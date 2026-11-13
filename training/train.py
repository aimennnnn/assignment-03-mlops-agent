"""Train one model and record everything about it in MLflow.

Examples:
    python -m training.train --model rf --n-estimators 100 --max-depth 5
    python -m training.train --model logreg --C 1.0 --register
"""

import argparse
import hashlib
import os
import subprocess

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = "data/cancer.csv"
TARGET = "target"
MODEL_NAME = "cancer-classifier"
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")


def git(*args):
    """Run a git command and return its output."""
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()


def dataset_provenance():
    """Work out which DVC dataset version the data file in data/ really is."""
    with open(DATA, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()  # same fingerprint DVC stores
    version, commit = "untracked", ""
    for tag in git("tag", "--sort=creatordate").split():
        if md5 in git("show", f"{tag}:{DATA}.dvc"):  # does this tag point at our data?
            version, commit = tag, git("rev-parse", "--short", tag + "^{commit}")
            break
    return {
        "dataset_path": DATA,
        "dataset_version": version,
        "dataset_git_commit": commit,
        "dataset_md5": md5,
        "code_git_commit": git("rev-parse", "--short", "HEAD"),
    }


def build_model(a):
    if a.model == "rf":
        params = {"n_estimators": a.n_estimators, "max_depth": a.max_depth}
        model = RandomForestClassifier(**params, random_state=a.seed)
    else:
        params = {"C": a.C}
        model = make_pipeline(StandardScaler(), LogisticRegression(C=a.C, max_iter=1000))
    return model, params


def compute_metrics(model, X_test, y_test):
    """Score a trained model on the test split."""
    pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
    }
    matrix = confusion_matrix(y_test, pred).tolist()
    return metrics, matrix


def log_to_mlflow(model, model_kind, params, metrics, matrix, example, rows,
                  register=False, extra_tags=None):
    """Record one training run in MLflow. Returns (run_id, model_uri, provenance)."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(f"cancer-{model_kind}")  # one experiment per model type
    with mlflow.start_run() as run:
        provenance = dataset_provenance()
        mlflow.set_tags(provenance)  # which data was used -> answers "which dataset version?"
        mlflow.set_tags(extra_tags or {})
        mlflow.log_params({"model": model_kind, **params})
        mlflow.log_param("dataset_rows", rows)
        mlflow.log_metrics(metrics)
        mlflow.log_dict(metrics, "metrics.json")
        mlflow.log_dict({"labels": [0, 1], "matrix": matrix}, "confusion_matrix.json")
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=example,
            skops_trusted_types=["sklearn.tree._tree.Tree"],  # needed to save tree models
            registered_model_name=MODEL_NAME if register else None,
        )
    return run.info.run_id, info.model_uri, provenance


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["rf", "logreg"], default="rf")
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--register", action="store_true", help="also add to the Model Registry")
    a = p.parse_args()

    df = pd.read_csv(DATA)
    X, y = df.drop(columns=[TARGET]).astype(float), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=a.test_size, random_state=a.seed, stratify=y
    )
    model, params = build_model(a)
    model.fit(X_train, y_train)
    metrics, matrix = compute_metrics(model, X_test, y_test)
    params = {**params, "test_size": a.test_size, "seed": a.seed}
    run_id, _, provenance = log_to_mlflow(
        model, a.model, params, metrics, matrix, X_test.head(2), len(df), register=a.register
    )

    print(f"run_id:   {run_id}")
    print(f"dataset:  {provenance['dataset_version']} ({len(df)} rows)")
    print("metrics:  " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
