"""Structured inputs and outputs of the agent.

Part 1: what the FastAPI interface accepts and returns.
Part 2: the arguments each tool accepts. The LLM's arguments are checked against
        these before a tool runs, so a bad or made-up argument is rejected.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ---------- Part 1: the conversational interface ----------


class PendingAction(BaseModel):
    """A state-changing action the agent wants to do but has NOT done yet."""

    tool: str
    arguments: dict[str, Any] = {}
    description: str = ""


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    # Send back the pending_action from the last reply to confirm it.
    confirm_action: PendingAction | None = None


class ToolCallRecord(BaseModel):
    """Evidence of one tool call: what ran, whether it worked, how long it took."""

    tool: str
    arguments: dict[str, Any] = {}
    status: str  # ok | error | invalid_arguments | needs_confirmation | timeout | unknown_tool
    ok: bool
    duration_ms: int
    error: str | None = None


class ChatResponse(BaseModel):
    request_id: str
    answer: str
    tools_used: list[ToolCallRecord] = []
    failed_tools: list[str] = []
    needs_confirmation: bool = False
    pending_action: PendingAction | None = None
    llm_calls: int = 0


# ---------- Part 2: tool arguments ----------


class Args(BaseModel):
    """Base class: unknown arguments are rejected."""

    model_config = ConfigDict(extra="forbid")


class NoArgs(Args):
    pass


class CompareDatasetVersions(Args):
    version_a: str = Field(description="older version, e.g. v1.0")
    version_b: str = Field(description="newer version, e.g. v2.0")


class DatasetVersion(Args):
    version: str = Field(description="dataset version (Git tag), e.g. v1.0")


class ListRuns(Args):
    experiment_name: str = Field("", description="empty = all experiments")
    max_results: int = Field(5, ge=1, le=20)


class RunId(Args):
    run_id: str = Field(description="full MLflow run id")


class FindBestRun(Args):
    metric: str = Field("f1", description="f1, accuracy, precision, recall or roc_auc")
    experiment_name: str = Field("", description="empty = all experiments")
    last_n: int = Field(5, ge=1, le=20, description="how many recent runs to compare")
    higher_is_better: bool = True


class CompareExperiments(Args):
    experiment_names: list[str] = Field([], description="empty = all experiments")
    metric: str = "f1"


class RegisteredModels(Args):
    model_name: str = Field("", description="empty = all registered models")


class DagRuns(Args):
    dag_id: str = "training_pipeline"
    limit: int = Field(5, ge=1, le=20)


class DagRunStatus(Args):
    dag_id: str = "training_pipeline"
    run_id: str = Field("", description="empty = the latest run")
    wait_seconds: int = Field(0, ge=0, le=120, description="wait for a running run to finish")


class TriggerDag(Args):
    dag_id: str = "training_pipeline"
    params: dict[str, Any] = Field(
        {}, description="optional: model (logreg|rf), n_estimators, max_depth, C, min_f1"
    )


class FailedTasks(Args):
    dag_id: str = "training_pipeline"
    run_id: str = Field("", description="empty = the latest run")


class TaskDiagnostics(Args):
    task_id: str
    dag_id: str = "training_pipeline"
    run_id: str = Field("", description="empty = the latest run")
    log_lines: int = Field(15, ge=1, le=50)


class PodName(Args):
    pod_name: str


class PodLogs(Args):
    pod_name: str
    tail_lines: int = Field(20, ge=1, le=100)
    previous: bool = Field(False, description="true = logs from before the last restart")


class DeploymentName(Args):
    deployment: str = "cancer-api"
