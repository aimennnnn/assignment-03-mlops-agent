"""Shared test helpers. No test needs a real service, a network or an API key."""

import dataclasses

import pytest

from agent import registry, tracing
from agent.tools import kubernetes_tools
from tests import fake_k8s


@pytest.fixture(autouse=True)
def logs_in_temp_folder(monkeypatch, tmp_path):
    """Send structured logs to a temp file so tests can read them back."""
    log_file = tmp_path / "agent.jsonl"
    monkeypatch.setattr(tracing, "LOG_FILE", log_file)
    return log_file


@pytest.fixture
def stub_tool(monkeypatch):
    """Replace a tool with a fake. Returns the list of calls the fake received."""

    def stub(name, result=None, func=None):
        calls = []

        def fake(**kwargs):
            calls.append(kwargs)
            if func:
                return func(**kwargs)
            return {"ok": True, **(result or {})}

        original = registry.BY_NAME[name]
        monkeypatch.setitem(registry.BY_NAME, name, dataclasses.replace(original, func=fake))
        return calls

    return stub


@pytest.fixture
def healthy_cluster():
    cluster = fake_k8s.FakeCluster(healthy=True)
    fake_k8s.install(kubernetes_tools, cluster)
    yield cluster
    kubernetes_tools._apis.clear()


@pytest.fixture
def broken_cluster():
    cluster = fake_k8s.FakeCluster(healthy=False)
    fake_k8s.install(kubernetes_tools, cluster)
    yield cluster
    kubernetes_tools._apis.clear()
