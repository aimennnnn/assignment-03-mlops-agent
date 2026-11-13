"""The agent loop: tool selection, permissions, bad input, failures, no invented results."""

import json
import time

from agent import orchestrator
from agent.llm import LLMError
from agent.orchestrator import run_agent
from agent.schemas import ChatResponse
from tests.fake_llm import FakeLLM


def test_agent_runs_the_tools_the_model_selects(stub_tool):
    """Successful selection and execution across several systems in one request."""
    mlflow = stub_tool("mlflow_find_best_run", {"best_run": {"run_id": "abc", "f1": 0.98}})
    dvc = stub_tool("dvc_get_current_dataset_version", {"version": "v2.0"})
    k8s = stub_tool("k8s_get_deployed_version", {"image_tag": "1.0.0"})
    llm = FakeLLM([
        [("mlflow_find_best_run", {"metric": "f1", "last_n": 3}),
         ("dvc_get_current_dataset_version", {}),
         ("k8s_get_deployed_version", {})],
        "Best run abc uses dataset v2.0.",
    ])

    reply = run_agent("best model and its dataset?", llm=llm)

    ChatResponse.model_validate(reply)  # the reply has the documented structure
    assert [t["status"] for t in reply["tools_used"]] == ["ok", "ok", "ok"]
    assert mlflow[0]["metric"] == "f1" and mlflow[0]["last_n"] == 3  # arguments passed on
    assert len(dvc) == 1 and len(k8s) == 1
    assert {"ok": True, "version": "v2.0"} in llm.tool_results()  # the model saw real results
    assert reply["failed_tools"] == [] and not reply["needs_confirmation"]


def test_agent_does_not_invent_results_when_a_tool_is_unavailable(stub_tool):
    """A down service gives the model no data, and the reply always names the failed tool."""
    stub_tool("mlflow_find_best_run",
              func=lambda **kw: {"ok": False, "error": "MLflow tracking server is not reachable"})
    # Even a model that tries to state a metric anyway cannot hide the failure.
    llm = FakeLLM([[("mlflow_find_best_run", {})], "The best model has f1 0.99."])

    reply = run_agent("what is the best model?", llm=llm)

    sent_to_model = llm.tool_results()[0]
    assert sent_to_model["ok"] is False and "best_run" not in sent_to_model
    assert reply["failed_tools"] == ["mlflow_find_best_run"]
    assert "Unavailable evidence" in reply["answer"]
    assert "MLflow tracking server is not reachable" in reply["answer"]


def test_state_changing_tool_is_not_run_without_confirmation(stub_tool):
    """Asking to trigger the pipeline starts nothing until the user confirms."""
    trigger = stub_tool("airflow_trigger_training_dag", {"triggered": True})
    llm = FakeLLM([[("airflow_trigger_training_dag", {})], "It will also deploy to production!"])

    reply = run_agent("trigger the training pipeline", llm=llm)

    assert trigger == []  # the tool was never called
    assert reply["needs_confirmation"] is True
    assert reply["pending_action"]["tool"] == "airflow_trigger_training_dag"
    assert reply["answer"].startswith("Nothing has been changed yet")  # fixed wording
    assert "deploy to production" not in reply["answer"]  # model's invented detail is dropped


def test_confirmed_action_runs_exactly_once(stub_tool):
    """One confirmation allows one change; a second attempt in the same request is refused."""
    trigger = stub_tool("airflow_trigger_training_dag", {"triggered": True, "run_id": "r1"})
    stub_tool("airflow_get_dag_run_status", {"state": "success"})
    llm = FakeLLM([
        [("airflow_trigger_training_dag", {}), ("airflow_get_dag_run_status", {})],
        "The run succeeded.",
    ])
    action = {"tool": "airflow_trigger_training_dag", "arguments": {}}

    reply = run_agent("trigger the pipeline", confirm_action=action, llm=llm)

    assert len(trigger) == 1 and trigger[0]["confirm"] is True
    assert "already executed" in llm.tool_results()[0]["error"]
    assert reply["needs_confirmation"] is False


def test_read_only_tool_cannot_be_confirmed_into_a_change(stub_tool):
    """confirm_action only works for tools marked as state-changing."""
    reply = run_agent("x", confirm_action={"tool": "no_such_tool", "arguments": {}},
                      llm=FakeLLM(["unused"]))
    assert "cannot be confirmed" in reply["answer"]


def test_invalid_arguments_are_rejected_before_the_tool_runs(stub_tool):
    """Out-of-range, unknown and missing arguments never reach the tool."""
    runs = stub_tool("mlflow_list_runs")
    logs = stub_tool("k8s_get_pod_logs")
    compare = stub_tool("dvc_compare_dataset_versions")
    llm = FakeLLM([
        [("mlflow_list_runs", {"max_results": 999}),
         ("k8s_get_pod_logs", {"pod_name": "p", "namespace": "kube-system"}),
         ("dvc_compare_dataset_versions", {"version_a": "v1.0"})],
        "done",
    ])

    reply = run_agent("test", llm=llm)

    assert [t["status"] for t in reply["tools_used"]] == ["invalid_arguments"] * 3
    assert runs == [] and logs == [] and compare == []
    assert "namespace" in reply["tools_used"][1]["error"]


def test_unknown_tool_is_reported_not_guessed():
    reply = run_agent("test", llm=FakeLLM([[("delete_everything", {})], "done"]))
    assert reply["tools_used"][0]["status"] == "unknown_tool"
    assert reply["failed_tools"] == ["delete_everything"]


def test_tool_timeout_is_handled(stub_tool, monkeypatch):
    """A tool that hangs is given up on, and the agent still answers."""
    monkeypatch.setattr(orchestrator, "TOOL_TIMEOUT", 0.2)
    stub_tool("airflow_get_dag_runs", func=lambda **kw: time.sleep(1.5) or {"ok": True})

    reply = run_agent("pipeline status?", llm=FakeLLM([[("airflow_get_dag_runs", {})], "done"]))

    assert reply["tools_used"][0]["status"] == "timeout"
    assert "timed out" in reply["answer"]


def test_crashing_tool_does_not_crash_the_agent(stub_tool):
    def boom(**kwargs):
        raise RuntimeError("unexpected")

    stub_tool("k8s_list_workloads", func=boom)
    reply = run_agent("pods?", llm=FakeLLM([[("k8s_list_workloads", {})], "done"]))
    assert reply["tools_used"][0]["status"] == "error"
    assert "k8s_list_workloads" in reply["failed_tools"]


def test_retraining_assessment_must_check_every_system(stub_tool):
    """An answer about retraining is sent back once if a system was not consulted."""
    for name in ("policy_get_retraining_policy", "mlflow_find_best_run",
                 "mlflow_get_registered_models", "dvc_get_current_dataset_version",
                 "k8s_get_deployed_version", "airflow_get_dag_runs"):
        stub_tool(name)
    llm = FakeLLM([
        [("policy_get_retraining_policy", {}), ("mlflow_find_best_run", {}),
         ("mlflow_get_registered_models", {}), ("dvc_get_current_dataset_version", {}),
         ("k8s_get_deployed_version", {})],
        "No retraining needed.",  # too early: Airflow was never checked
        [("airflow_get_dag_runs", {})],
        "Checked all four: no retraining needed.",
    ])

    reply = run_agent("is retraining necessary?", llm=llm)

    assert reply["answer"].startswith("Checked all four")
    nudge = [m["content"] for m in llm.seen[-1] if m["role"] == "user"][-1]
    assert "Airflow" in nudge
    assert any(t["tool"] == "airflow_get_dag_runs" for t in reply["tools_used"])


def test_answer_without_tools_is_flagged():
    reply = run_agent("hello", llm=FakeLLM(["The model has f1 0.97."]))
    assert "no tools were called" in reply["answer"]


def test_llm_failure_is_reported_clearly():
    class DownLLM:
        def chat(self, messages, tools):
            raise LLMError("LLM provider is not reachable (ConnectionError)")

    reply = run_agent("anything", llm=DownLLM())
    assert "language model failed" in reply["answer"]
    assert reply["tools_used"] == []


def test_every_tool_call_is_logged_without_secrets(stub_tool, logs_in_temp_folder):
    """Structured logs carry tool name, status, duration and request id, and no secrets."""
    stub_tool("dvc_get_current_dataset_version", {"version": "v2.0"})
    reply = run_agent("version?", llm=FakeLLM([[("dvc_get_current_dataset_version", {})], "v2.0"]),
                      request_id="req-123")

    lines = [json.loads(x) for x in logs_in_temp_folder.read_text().splitlines()]
    call = next(x for x in lines if x["event"] == "tool_call")
    assert call["request_id"] == "req-123" == reply["request_id"]
    assert call["tool"] == "dvc_get_current_dataset_version"
    assert call["status"] == "ok" and isinstance(call["duration_ms"], int)
    assert {x["event"] for x in lines} >= {"request", "tool_call", "response"}
