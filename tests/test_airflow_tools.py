"""Airflow tools with the HTTP layer mocked: approvals, outages and failed runs."""

import pytest
import requests

from agent.tools import airflow_tools as af


class Reply:
    def __init__(self, data, status=200):
        self._data, self.status_code, self.text = data, status, str(data)

    def json(self):
        return self._data


@pytest.fixture
def airflow(monkeypatch):
    """A fake Airflow API. Returns the list of requests the tools made."""
    sent = []
    runs = [
        {"dag_run_id": "run-2", "state": "failed", "start_date": "2026-11-14T10:00:00Z",
         "end_date": "2026-11-14T10:00:30Z", "duration": 30.0, "triggered_by": "rest_api",
         "conf": {"model": "rf"}},
        {"dag_run_id": "run-1", "state": "success", "start_date": "2026-11-14T09:00:00Z",
         "end_date": "2026-11-14T09:00:40Z", "duration": 40.0, "triggered_by": "rest_api",
         "conf": {}},
    ]
    tasks = [
        {"task_id": "train_model", "state": "success", "try_number": 1, "max_tries": 2,
         "start_date": "2026-11-14T10:00:10Z", "duration": 3.0},
        {"task_id": "apply_quality_gate", "state": "failed", "try_number": 1, "max_tries": 0,
         "start_date": "2026-11-14T10:00:20Z", "duration": 2.0},
    ]

    def request(method, url, **kwargs):
        sent.append((method, url))
        if url.endswith("/taskInstances"):
            return Reply({"task_instances": tasks})
        if method == "POST" and url.endswith("/dagRuns"):
            return Reply({**runs[1], "dag_run_id": "run-3", "state": "queued"})
        if url.endswith("/dagRuns"):
            return Reply({"dag_runs": runs})
        for run in runs:
            if url.endswith("/dagRuns/" + run["dag_run_id"]):
                return Reply(run)
        return Reply({"detail": "not found"}, 404)

    monkeypatch.setattr(af, "_token", "test-token")
    monkeypatch.setattr(af.requests, "request", request)
    return sent


def test_unapproved_dag_is_never_triggered(airflow):
    result = af.trigger_training_dag("drop_all_tables", confirm=True)
    assert result["ok"] is False and "not approved" in result["error"]
    assert airflow == []  # no request reached Airflow


def test_trigger_needs_confirmation(airflow):
    asked = af.trigger_training_dag()
    assert asked["needs_confirmation"] is True and airflow == []

    started = af.trigger_training_dag(confirm=True)
    assert started["triggered"] is True and started["run_id"] == "run-3"
    assert airflow[0][0] == "POST"


def test_failed_run_names_the_failed_task(airflow):
    """Injected failure: the quality gate rejected a weak model."""
    runs = af.get_dag_runs(limit=2)["runs"]
    assert runs[0]["state"] == "failed" and runs[0]["failed_tasks"] == ["apply_quality_gate"]
    assert "failed_tasks" not in runs[1]

    failed = af.get_failed_tasks()
    assert [t["task_id"] for t in failed["failed_tasks"]] == ["apply_quality_gate"]


def test_airflow_down_returns_an_error_not_data(monkeypatch):
    """Service failure: Airflow is not running."""
    def refuse(*args, **kwargs):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(af, "_token", "test-token")
    monkeypatch.setattr(af.requests, "request", refuse)
    result = af.get_dag_runs()
    assert result["ok"] is False and "not reachable" in result["error"]
    assert "runs" not in result


def test_airflow_timeout(monkeypatch):
    def slow(*args, **kwargs):
        raise requests.exceptions.Timeout("too slow")

    monkeypatch.setattr(af, "_token", "test-token")
    monkeypatch.setattr(af.requests, "request", slow)
    result = af.list_dags()
    assert result["ok"] is False and "did not answer" in result["error"]


def test_bad_dag_id_is_rejected(airflow):
    result = af.get_dag_runs("../../etc/passwd")
    assert result["ok"] is False and airflow == []
