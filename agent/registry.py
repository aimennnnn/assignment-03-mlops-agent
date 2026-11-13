"""The tool registry: every tool the agent may use, with its permissions.

Each row is: name the LLM sees, the Python function, its argument schema,
a one-line description, and whether it changes state (needs user confirmation).
"""

from dataclasses import dataclass
from typing import Callable

from agent import schemas as s
from agent.tools import airflow_tools as af
from agent.tools import dvc_tools as dvc
from agent.tools import kubernetes_tools as k8s
from agent.tools import mlflow_tools as ml
from agent.tools import policy_tools as policy


@dataclass(frozen=True)
class Tool:
    name: str
    func: Callable
    args: type[s.Args]
    description: str
    changes_state: bool = False  # True = never runs without the user's confirmation


TOOLS = [
    # --- DVC: dataset versions ---
    Tool("dvc_list_tracked_datasets", dvc.list_tracked_datasets, s.NoArgs,
         "List datasets tracked by DVC with their hash and size."),
    Tool("dvc_get_current_dataset_version", dvc.get_current_dataset_version, s.NoArgs,
         "Current dataset version (Git tag), Git commit, data hash and DVC remote."),
    Tool("dvc_compare_dataset_versions", dvc.compare_dataset_versions, s.CompareDatasetVersions,
         "Compare two dataset versions (hash and size change)."),
    Tool("dvc_check_data_changed", dvc.check_data_changed, s.NoArgs,
         "Detect whether tracked data was modified since it was last saved in DVC."),
    Tool("dvc_explain_reproduce_version", dvc.explain_reproduce_version, s.DatasetVersion,
         "Steps to restore a previous dataset version."),
    # --- MLflow: experiments and models ---
    Tool("mlflow_list_experiments", ml.list_experiments, s.NoArgs,
         "List MLflow experiments and their run counts."),
    Tool("mlflow_list_runs", ml.list_runs, s.ListRuns,
         "Most recent training runs with metrics, params and dataset version."),
    Tool("mlflow_get_run_details", ml.get_run_details, s.RunId,
         "One run's metrics, params, artifacts and the dataset version it was trained on."),
    Tool("mlflow_find_best_run", ml.find_best_run, s.FindBestRun,
         "Rank recent runs by a metric and return the best one, with cautions."),
    Tool("mlflow_compare_experiments", ml.compare_experiments, s.CompareExperiments,
         "Compare experiments: best, average and worst of a metric."),
    Tool("mlflow_get_registered_models", ml.get_registered_models, s.RegisteredModels,
         "Model Registry versions, aliases (champion = approved model), source run, metrics."),
    # --- Airflow: training pipeline ---
    Tool("airflow_list_dags", af.list_dags, s.NoArgs,
         "List Airflow DAGs and DAG files that failed to load."),
    Tool("airflow_get_dag_runs", af.get_dag_runs, s.DagRuns,
         "Recent runs of a DAG with state and duration."),
    Tool("airflow_get_dag_run_status", af.get_dag_run_status, s.DagRunStatus,
         "State of one DAG run and of each of its tasks."),
    Tool("airflow_get_failed_tasks", af.get_failed_tasks, s.FailedTasks,
         "Which tasks failed in a run and which were skipped because of it."),
    Tool("airflow_get_task_diagnostics", af.get_task_diagnostics, s.TaskDiagnostics,
         "One task's tries, timing and last log lines, for diagnosing a failure."),
    Tool("airflow_trigger_training_dag", af.trigger_training_dag, s.TriggerDag,
         "Start one run of the training DAG (validate data, preprocess, train, evaluate and "
         "log to MLflow, quality gate; only a model that beats the champion is promoted in "
         "the registry). It does not deploy anything. Only when the user asks to trigger.",
         changes_state=True),
    # --- Kubernetes: the deployed prediction API ---
    Tool("k8s_list_workloads", k8s.list_workloads, s.NoArgs,
         "Deployments and Pods of the prediction API with readiness and restarts."),
    Tool("k8s_get_pod_status", k8s.get_pod_status, s.PodName,
         "One Pod's conditions, container state, probes and events."),
    Tool("k8s_get_pod_logs", k8s.get_pod_logs, s.PodLogs,
         "Last log lines of a Pod, with error lines picked out."),
    Tool("k8s_get_services", k8s.get_services, s.NoArgs,
         "Services and how many ready Pods each can reach."),
    Tool("k8s_find_unhealthy_pods", k8s.find_unhealthy_pods, s.NoArgs,
         "Pods that are not ready or restarted, with reasons and restart events."),
    Tool("k8s_get_deployed_version", k8s.get_deployed_version, s.DeploymentName,
         "Deployed app version, image, config and the model version the app reports."),
    Tool("k8s_restart_deployment", k8s.restart_deployment, s.DeploymentName,
         "Restart the Deployment's pods one by one (rolling restart, same image and "
         "config). Only when the user explicitly asks to restart.",
         changes_state=True),
    # --- Policy ---
    Tool("policy_get_retraining_policy", policy.get_retraining_policy, s.NoArgs,
         "The explicit rules for when retraining may be recommended. Call before advising."),
]

BY_NAME = {t.name: t for t in TOOLS}


def _clean(schema):
    """Remove fields the LLM does not need, to keep the tool list small."""
    if isinstance(schema, dict):
        return {k: _clean(v) for k, v in schema.items() if k != "title"}
    if isinstance(schema, list):
        return [_clean(v) for v in schema]
    return schema


def llm_tool_specs():
    """The tool list in the standard chat-API format (name, description, JSON schema)."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": _clean(t.args.model_json_schema()),
            },
        }
        for t in TOOLS
    ]


def describe_tools():
    """Human-readable list of tools and permissions (shown at GET /tools)."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "permission": "needs user confirmation" if t.changes_state else "read-only",
            "arguments": list(t.args.model_fields),
        }
        for t in TOOLS
    ]
