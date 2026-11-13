"""Training pipeline rules: bad data and weak models must never be promoted."""

import hashlib
import json
from types import SimpleNamespace

import mlflow
import pandas as pd
import pytest

from training import pipeline


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    """A small valid dataset with a matching DVC pointer file."""
    path = tmp_path / "cancer.csv"
    pd.DataFrame({"a": range(120), "target": [0, 1] * 60}).to_csv(path, index=False)
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    (tmp_path / "cancer.csv.dvc").write_text(f"outs:\n- md5: {md5}\n  path: cancer.csv\n")
    monkeypatch.setattr(pipeline, "DATA", str(path))
    return path


def test_validation_passes_for_tracked_data(dataset, tmp_path):
    pipeline.validate(None, tmp_path)
    assert json.loads((tmp_path / "validation.json").read_text())["rows"] == 120


def test_validation_fails_when_data_changed_without_dvc(dataset, tmp_path):
    """Injected failure: someone edits the data file but does not run dvc add."""
    with dataset.open("a") as f:
        f.write("999,1\n")
    with pytest.raises(SystemExit):
        pipeline.validate(None, tmp_path)
    assert not (tmp_path / "validation.json").exists()  # later stages get nothing to work on


class FakeRegistry:
    """Stands in for MLflow in the quality-gate tests."""

    def __init__(self, champion_f1=None):
        self.champion_f1, self.tags, self.aliases, self.registered = champion_f1, {}, {}, []

    def get_model_version_by_alias(self, name, alias):
        if self.champion_f1 is None:
            raise mlflow.exceptions.MlflowException("no champion")
        return SimpleNamespace(version="4", run_id="champion-run")

    def get_run(self, run_id):
        return SimpleNamespace(data=SimpleNamespace(metrics={"f1": self.champion_f1}))

    def set_tag(self, run_id, key, value):
        self.tags[key] = value

    def set_registered_model_alias(self, name, alias, version):
        self.aliases[alias] = version


def run_gate(tmp_path, monkeypatch, f1, champion_f1, min_f1=0.95):
    registry = FakeRegistry(champion_f1)
    monkeypatch.setattr(mlflow, "MlflowClient", lambda: registry)
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(mlflow, "register_model", lambda uri, name: (
        registry.registered.append(uri) or SimpleNamespace(version="5")))
    (tmp_path / "evaluation.json").write_text(json.dumps(
        {"mlflow_run_id": "new-run", "model_uri": "models:/m-1", "metrics": {"f1": f1}}))
    pipeline.gate(SimpleNamespace(min_f1=min_f1), tmp_path)
    return registry


def test_gate_blocks_a_model_below_the_threshold(tmp_path, monkeypatch):
    registry = FakeRegistry(0.98)
    monkeypatch.setattr(mlflow, "MlflowClient", lambda: registry)
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)
    (tmp_path / "evaluation.json").write_text(json.dumps(
        {"mlflow_run_id": "new-run", "model_uri": "models:/m-1", "metrics": {"f1": 0.90}}))

    with pytest.raises(SystemExit):
        pipeline.gate(SimpleNamespace(min_f1=0.95), tmp_path)

    assert registry.tags["quality_gate"] == "failed"
    assert registry.registered == [] and registry.aliases == {}  # nothing promoted


def test_gate_blocks_a_model_worse_than_the_champion(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        run_gate(tmp_path, monkeypatch, f1=0.96, champion_f1=0.98)


def test_gate_keeps_the_champion_when_the_new_model_only_matches_it(tmp_path, monkeypatch):
    registry = run_gate(tmp_path, monkeypatch, f1=0.98, champion_f1=0.98)
    assert registry.tags == {"quality_gate": "passed", "promoted": "false"}
    assert registry.registered == [] and registry.aliases == {}


def test_gate_promotes_a_better_model(tmp_path, monkeypatch):
    registry = run_gate(tmp_path, monkeypatch, f1=0.99, champion_f1=0.98)
    assert registry.aliases == {"champion": "5"} and registry.tags["promoted"] == "true"


def test_gate_promotes_the_first_good_model(tmp_path, monkeypatch):
    registry = run_gate(tmp_path, monkeypatch, f1=0.97, champion_f1=None)
    assert registry.aliases == {"champion": "5"}
