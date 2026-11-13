"""Copy the champion model out of the MLflow Model Registry into api/model/.

    python -m training.export_model

The prediction API image is then built with this exact model inside it.
model_info.json records where the model came from (version, run, dataset).
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import sklearn
from mlflow import MlflowClient

from training.train import MODEL_NAME, TRACKING_URI

ALIAS = "champion"
OUT = Path("api/model")


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    version = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    run = client.get_run(version.run_id)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{ALIAS}")

    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT / "model.joblib")
    info = {
        "model_name": MODEL_NAME,
        "model_version": version.version,
        "alias": ALIAS,
        "mlflow_run_id": version.run_id,
        "f1": round(run.data.metrics.get("f1", 0), 4),
        "dataset_version": run.data.tags.get("dataset_version"),
        "dataset_git_commit": run.data.tags.get("dataset_git_commit"),
        "n_features": int(model.n_features_in_),
        "sklearn_version": sklearn.__version__,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (OUT / "model_info.json").write_text(json.dumps(info, indent=2))
    print(f"Exported {MODEL_NAME} version {version.version} ({ALIAS}) to {OUT}")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
