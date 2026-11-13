"""Task 2 - MLflow experiment intelligence tools.

Each function below is one "tool" the AI agent can call.
Every tool is read-only: it asks the MLflow tracking server through
MlflowClient and returns what the server recorded. Nothing is made up.

Try it by hand:   python -m agent.tools.mlflow_tools
"""

import json
import os
import re
from datetime import datetime, timezone

# If the server is down, fail in a few seconds instead of retrying for minutes.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")

from mlflow import MlflowClient  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

# A metric name must look like f1 or roc_auc.
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def _client():
    return MlflowClient(tracking_uri=TRACKING_URI)


def _safe(tool):
    """Turn any failure into {"ok": False, "error": ...} so the agent can report it."""

    def wrapper(*args, **kwargs):
        try:
            return {"ok": True, "tool": tool.__name__, **tool(*args, **kwargs)}
        except ValueError as e:
            return {"ok": False, "tool": tool.__name__, "error": str(e)}
        except MlflowException as e:
            text = str(e)
            if "RESOURCE_DOES_NOT_EXIST" in text or "not found" in text.lower():
                error = "Not found in MLflow: " + text[:200]
            else:
                error = f"MLflow tracking server at {TRACKING_URI} is not reachable or failed"
            return {"ok": False, "tool": tool.__name__, "error": error}
        except Exception as e:  # connection refused, timeout, ...
            kind = type(e).__name__
            error = f"MLflow tracking server at {TRACKING_URI} is not reachable ({kind})"
            return {"ok": False, "tool": tool.__name__, "error": error}

    wrapper.__name__ = tool.__name__
    wrapper.__doc__ = tool.__doc__
    return wrapper


def _experiment_ids(client, experiment_name=None):
    """Turn an experiment name into its id (or return all experiment ids)."""
    experiments = client.search_experiments()
    if experiment_name is None:
        return [e.experiment_id for e in experiments]
    for e in experiments:
        if e.name == experiment_name:
            return [e.experiment_id]
    names = [e.name for e in experiments]
    raise ValueError(f"Experiment '{experiment_name}' does not exist. Available: {names}")


def _run_summary(run, names):
    """The useful facts about one run, as a plain dict."""
    tags = run.data.tags
    return {
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,
        "experiment": names.get(run.info.experiment_id, run.info.experiment_id),
        "status": run.info.status,
        "started": datetime.fromtimestamp(run.info.start_time / 1000, timezone.utc).isoformat(
            timespec="seconds"
        ),
        "metrics": {k: round(v, 4) for k, v in run.data.metrics.items()},
        "params": dict(run.data.params),
        "dataset_version": tags.get("dataset_version"),
        "dataset_git_commit": tags.get("dataset_git_commit"),
    }


def _recent_runs(client, experiment_name, max_results):
    names = {e.experiment_id: e.name for e in client.search_experiments()}
    runs = client.search_runs(
        _experiment_ids(client, experiment_name),
        order_by=["attributes.start_time DESC"],
        max_results=max_results,
    )
    return [_run_summary(r, names) for r in runs]


@_safe
def list_experiments():
    """List all MLflow experiments and how many runs each has."""
    client = _client()
    out = []
    for e in client.search_experiments():
        runs = client.search_runs([e.experiment_id], max_results=1000)
        out.append({"experiment_id": e.experiment_id, "name": e.name, "run_count": len(runs)})
    return {"count": len(out), "experiments": out}


@_safe
def list_runs(experiment_name=None, max_results=5):
    """List the most recent training runs (newest first), in one experiment or in all."""
    runs = _recent_runs(_client(), experiment_name, int(max_results))
    return {"count": len(runs), "runs": runs}


@_safe
def get_run_details(run_id):
    """Get one run's metrics, parameters, artifacts and the dataset version it used."""
    client = _client()
    run = client.get_run(run_id)
    names = {e.experiment_id: e.name for e in client.search_experiments()}
    tags = run.data.tags
    outputs = getattr(run, "outputs", None)
    return {
        **_run_summary(run, names),
        "dataset": {k: v for k, v in tags.items() if k.startswith("dataset_")},
        "code_git_commit": tags.get("code_git_commit"),
        "artifacts": [a.path for a in client.list_artifacts(run_id)],
        "logged_models": [m.model_id for m in outputs.model_outputs] if outputs else [],
    }


@_safe
def find_best_run(metric="f1", experiment_name=None, last_n=5, higher_is_better=True):
    """Compare the last N runs on one metric and say which is best, with cautions."""
    if not SAFE_NAME.match(metric or ""):
        raise ValueError(f"'{metric}' is not a valid metric name")
    client = _client()
    runs = _recent_runs(client, experiment_name, int(last_n))
    scored = [r for r in runs if metric in r["metrics"]]
    if not scored:
        seen = sorted({m for r in runs for m in r["metrics"]})
        raise ValueError(f"No recent run has metric '{metric}'. Recorded metrics: {seen}")
    scored.sort(key=lambda r: r["metrics"][metric], reverse=bool(higher_is_better))
    best = scored[0]

    # Facts that stop the agent from over-claiming. Each one comes from recorded data.
    cautions = []
    versions = sorted({str(r["dataset_version"]) for r in scored})
    if len(versions) > 1:
        cautions.append(
            f"Runs used different dataset versions {versions}, so scores are not "
            "measured on the same test data."
        )
    if len(scored) > 1:
        gap = abs(best["metrics"][metric] - scored[1]["metrics"][metric])
        cautions.append(f"Lead over the second-best run is {gap:.4f} on a single test split.")
    registered = [
        f"{v.name} v{v.version}"
        for v in client.search_model_versions(f"run_id='{best['run_id']}'")
    ]
    cautions.append(
        f"Best run is registered as {registered}." if registered
        else "Best run is NOT in the Model Registry."
    )
    cautions.append(
        f"Best {metric} alone does not make a model production-ready: "
        "nothing here checks latency, fairness, or behaviour on new data."
    )
    return {
        "metric": metric,
        "higher_is_better": bool(higher_is_better),
        "runs_compared": len(scored),
        "ranking": scored,
        "best_run": best,
        "cautions": cautions,
    }


@_safe
def compare_experiments(experiment_names=None, metric="f1"):
    """Compare two or more experiments: run count, best and average of one metric."""
    if not SAFE_NAME.match(metric or ""):
        raise ValueError(f"'{metric}' is not a valid metric name")
    client = _client()
    if not experiment_names:
        experiment_names = [e.name for e in client.search_experiments() if e.name != "Default"]
    out = []
    for name in experiment_names:
        runs = [r for r in _recent_runs(client, name, 1000) if metric in r["metrics"]]
        values = [r["metrics"][metric] for r in runs]
        top = max(runs, key=lambda r: r["metrics"][metric]) if runs else None
        out.append(
            {
                "experiment": name,
                "runs_with_metric": len(runs),
                "best": max(values) if values else None,
                "average": round(sum(values) / len(values), 4) if values else None,
                "worst": min(values) if values else None,
                "best_run_id": top and top["run_id"],
                "best_run_params": top and top["params"],
                "best_run_dataset_version": top and top["dataset_version"],
            }
        )
    return {"metric": metric, "experiments": out}


@_safe
def get_registered_models(model_name=None):
    """Get Model Registry information: model versions, their source run and metrics."""
    client = _client()
    if model_name and not re.match(r"^[A-Za-z0-9_. -]+$", model_name):
        raise ValueError(f"'{model_name}' is not a valid model name")
    query = f"name='{model_name}'" if model_name else None
    versions = []
    aliases = {}  # (model name, version) -> alias names such as "champion"
    for m in client.search_registered_models():
        for alias, version in (m.aliases or {}).items():
            aliases.setdefault((m.name, str(version)), []).append(alias)
    for v in client.search_model_versions(query):
        metrics = {}
        if v.run_id:
            metrics = {k: round(x, 4) for k, x in client.get_run(v.run_id).data.metrics.items()}
        versions.append(
            {
                "name": v.name,
                "version": v.version,
                "status": v.status,
                "aliases": aliases.get((v.name, str(v.version)), []),
                "run_id": v.run_id,
                "created": datetime.fromtimestamp(
                    v.creation_timestamp / 1000, timezone.utc
                ).isoformat(timespec="seconds"),
                "run_metrics": metrics,
            }
        )
    versions.sort(key=lambda v: (v["name"], int(v["version"])))
    note = None if versions else "No registered model versions were found."
    return {"count": len(versions), "model_versions": versions, "note": note}


# The list the agent will pick from in Step 5.
MLFLOW_TOOLS = [
    list_experiments,
    list_runs,
    get_run_details,
    find_best_run,
    compare_experiments,
    get_registered_models,
]


if __name__ == "__main__":
    best = find_best_run("f1", last_n=5)
    best_id = best["best_run"]["run_id"] if best["ok"] else "none"
    for result in (
        list_experiments(),
        list_runs(max_results=2),
        best,
        get_run_details(best_id),
        compare_experiments(),
        get_registered_models(),
        get_run_details("0000"),  # does not exist -> clean error, no guess
        find_best_run("made_up_metric"),  # not recorded -> clean error, no guess
    ):
        print(json.dumps(result, indent=2), "\n")
