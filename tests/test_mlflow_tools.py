"""MLflow tools with the tracking client mocked: best-run logic, cautions and outages."""

from types import SimpleNamespace

import pytest

from agent.tools import mlflow_tools as ml


def make_run(run_id, f1, dataset, started):
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id, run_name=run_id, experiment_id="1",
                             status="FINISHED", start_time=started),
        data=SimpleNamespace(metrics={"f1": f1}, params={"model": "rf"},
                             tags={"dataset_version": dataset, "dataset_git_commit": "c986c4e"}),
    )


class FakeMlflow:
    def __init__(self):
        self.runs = [make_run("new", 0.95, "v2.0", 3000), make_run("best", 0.98, "v2.0", 2000),
                     make_run("old", 0.97, "v1.0", 1000)]

    def search_experiments(self):
        return [SimpleNamespace(experiment_id="1", name="cancer-rf")]

    def search_runs(self, experiment_ids, order_by=None, max_results=1000, **kwargs):
        return self.runs[:max_results]

    def search_model_versions(self, query=None):
        return []

    def search_registered_models(self):
        return []


@pytest.fixture
def mlflow(monkeypatch):
    fake = FakeMlflow()
    monkeypatch.setattr(ml, "_client", lambda: fake)
    return fake


def test_best_run_is_chosen_by_the_metric(mlflow):
    result = ml.find_best_run("f1", last_n=3)
    assert result["best_run"]["run_id"] == "best"
    assert [r["run_id"] for r in result["ranking"]] == ["best", "old", "new"]


def test_best_run_comes_with_cautions(mlflow):
    """The highest F1 is not presented as production-ready."""
    cautions = " ".join(ml.find_best_run("f1", last_n=3)["cautions"])
    assert "different dataset versions" in cautions
    assert "NOT in the Model Registry" in cautions
    assert "does not make a model production-ready" in cautions


def test_metric_that_was_never_recorded_is_an_error(mlflow):
    result = ml.find_best_run("made_up_metric")
    assert result["ok"] is False and "No recent run has metric" in result["error"]


def test_unknown_experiment_is_an_error(mlflow):
    result = ml.list_runs("no-such-experiment")
    assert result["ok"] is False and "does not exist" in result["error"]


def test_empty_registry_is_reported_as_empty(mlflow):
    result = ml.get_registered_models()
    assert result["count"] == 0 and result["champion"] is None


def test_mlflow_down_returns_an_error_not_data(monkeypatch):
    """Service failure: the tracking server is not running."""
    class Down:
        def search_experiments(self):
            raise ConnectionError("refused")

    monkeypatch.setattr(ml, "_client", lambda: Down())
    result = ml.list_experiments()
    assert result["ok"] is False and "not reachable" in result["error"]
    assert "experiments" not in result
