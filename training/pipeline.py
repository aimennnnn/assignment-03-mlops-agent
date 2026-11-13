"""The five stages of the training pipeline. Airflow runs one stage per task.

    python -m training.pipeline validate   --run-dir runs/demo
    python -m training.pipeline preprocess --run-dir runs/demo
    python -m training.pipeline train      --run-dir runs/demo --model logreg
    python -m training.pipeline evaluate   --run-dir runs/demo --model logreg
    python -m training.pipeline gate       --run-dir runs/demo --min-f1 0.95

Each stage reads what the stage before it saved in --run-dir.
A stage that fails exits with an error, so Airflow stops the stages after it.
"""

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from training.train import (
    DATA,
    MODEL_NAME,
    TARGET,
    TRACKING_URI,
    build_model,
    compute_metrics,
    log_to_mlflow,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)
log = logging.getLogger("pipeline")

MIN_ROWS = 100
CHAMPION = "champion"  # registry alias of the model that is allowed to be deployed


def fail(message):
    """Stop this stage with an error. Airflow marks the task as failed."""
    log.error("FAILED: %s", message)
    sys.exit(1)


def validate(a, run_dir):
    """Stage 1 - is the dataset safe to train on?"""
    if not Path(DATA).exists():
        fail(f"{DATA} does not exist")
    with open(DATA, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    pointer = Path(DATA + ".dvc")
    tracked = re.search(r"md5:\s*(\w+)", pointer.read_text()) if pointer.exists() else None
    if not tracked:
        fail(f"{DATA} is not tracked by DVC (no {pointer})")
    if tracked.group(1) != md5:
        fail(f"{DATA} was changed without 'dvc add': its hash {md5} is not the DVC "
             f"version {tracked.group(1)}")

    df = pd.read_csv(DATA)
    if TARGET not in df.columns:
        fail(f"column '{TARGET}' is missing")
    if len(df) < MIN_ROWS:
        fail(f"only {len(df)} rows, need at least {MIN_ROWS}")
    if df.isna().any().any():
        fail(f"{int(df.isna().sum().sum())} missing values found")
    if set(df[TARGET].unique()) != {0, 1}:
        fail(f"'{TARGET}' must contain both 0 and 1, found {sorted(df[TARGET].unique())}")

    report = {"rows": len(df), "columns": len(df.columns), "dataset_md5": md5}
    (run_dir / "validation.json").write_text(json.dumps(report))
    log.info("validation passed: %s", report)


def preprocess(a, run_dir):
    """Stage 2 - split the data into a training part and a test part."""
    df = pd.read_csv(DATA)
    train, test = train_test_split(
        df, test_size=a.test_size, random_state=a.seed, stratify=df[TARGET]
    )
    train.to_csv(run_dir / "train.csv", index=False)
    test.to_csv(run_dir / "test.csv", index=False)
    log.info("split %d rows into %d train and %d test", len(df), len(train), len(test))


def train(a, run_dir):
    """Stage 3 - fit the model on the training part."""
    df = pd.read_csv(run_dir / "train.csv")
    model, params = build_model(a)
    model.fit(df.drop(columns=[TARGET]).astype(float), df[TARGET])
    joblib.dump(model, run_dir / "model.joblib")
    log.info("trained %s with %s on %d rows", a.model, params, len(df))


def evaluate(a, run_dir):
    """Stage 4 - score the model on the test part and record everything in MLflow."""
    test = pd.read_csv(run_dir / "test.csv")
    rows = len(test) + len(pd.read_csv(run_dir / "train.csv"))
    model = joblib.load(run_dir / "model.joblib")
    X, y = test.drop(columns=[TARGET]).astype(float), test[TARGET]
    metrics, matrix = compute_metrics(model, X, y)
    _, params = build_model(a)
    params = {**params, "test_size": a.test_size, "seed": a.seed}
    run_id, model_uri, provenance = log_to_mlflow(
        model, a.model, params, metrics, matrix, X.head(2), rows,
        extra_tags={"pipeline": "airflow", "pipeline_run": a.pipeline_run},
    )
    result = {"mlflow_run_id": run_id, "model_uri": model_uri, "metrics": metrics,
              "dataset_version": provenance["dataset_version"]}
    (run_dir / "evaluation.json").write_text(json.dumps(result))
    log.info("logged MLflow run %s on dataset %s", run_id, provenance["dataset_version"])
    log.info("metrics: %s", {k: round(v, 4) for k, v in metrics.items()})


def gate(a, run_dir):
    """Stage 5 - only a model that is good enough gets promoted in the Model Registry."""
    import mlflow
    from mlflow import MlflowClient

    result = json.loads((run_dir / "evaluation.json").read_text())
    run_id, f1 = result["mlflow_run_id"], result["metrics"]["f1"]
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    reason = None
    if f1 < a.min_f1:
        reason = f"f1 {f1:.4f} is below the required {a.min_f1}"
    else:
        try:
            current = client.get_model_version_by_alias(MODEL_NAME, CHAMPION)
            best = client.get_run(current.run_id).data.metrics.get("f1", 0)
            if f1 < best:
                reason = (f"f1 {f1:.4f} is worse than the current {CHAMPION} "
                          f"(version {current.version}, f1 {best:.4f})")
        except mlflow.exceptions.MlflowException:
            log.info("no %s model yet, so there is nothing to beat", CHAMPION)

    if reason:
        client.set_tag(run_id, "quality_gate", "failed")
        client.set_tag(run_id, "quality_gate_reason", reason)
        fail(f"quality gate: {reason}. Model was NOT promoted.")

    version = mlflow.register_model(result["model_uri"], MODEL_NAME).version
    client.set_registered_model_alias(MODEL_NAME, CHAMPION, version)
    client.set_tag(run_id, "quality_gate", "passed")
    log.info("quality gate passed (f1 %.4f >= %s)", f1, a.min_f1)
    log.info("promoted: %s version %s is now '%s'", MODEL_NAME, version, CHAMPION)


STAGES = {"validate": validate, "preprocess": preprocess, "train": train,
          "evaluate": evaluate, "gate": gate}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=STAGES)
    p.add_argument("--run-dir", required=True, help="folder shared by the stages of one run")
    p.add_argument("--pipeline-run", default="manual", help="Airflow run id, saved as a tag")
    p.add_argument("--model", choices=["rf", "logreg"], default="logreg")
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-f1", type=float, default=0.95)
    a = p.parse_args()

    run_dir = Path(a.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("stage '%s' started", a.stage)
    STAGES[a.stage](a, run_dir)
    log.info("stage '%s' finished", a.stage)


if __name__ == "__main__":
    main()
