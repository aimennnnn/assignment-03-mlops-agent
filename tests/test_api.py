"""The FastAPI interface: request validation and structured replies."""

from fastapi.testclient import TestClient

from agent import main, orchestrator
from tests.fake_llm import FakeLLM

client = TestClient(main.app)


def test_health_and_tool_list():
    assert client.get("/health").json() == {"status": "ok"}
    tools = client.get("/tools").json()
    assert tools["count"] == 25
    changing = [t["name"] for t in tools["tools"] if t["permission"] != "read-only"]
    assert changing == ["airflow_trigger_training_dag", "k8s_restart_deployment"]


def test_invalid_requests_are_rejected():
    assert client.post("/chat", json={}).status_code == 422
    assert client.post("/chat", json={"message": ""}).status_code == 422
    assert client.post("/chat", json={"message": "x" * 5000}).status_code == 422


def test_chat_returns_a_structured_reply(stub_tool, monkeypatch):
    stub_tool("dvc_get_current_dataset_version", {"version": "v2.0"})
    monkeypatch.setattr(orchestrator, "LLM", lambda: FakeLLM(
        [[("dvc_get_current_dataset_version", {})], "The current dataset version is v2.0."]))

    reply = client.post("/chat", json={"message": "Which dataset version is current?"})

    body = reply.json()
    assert reply.status_code == 200
    assert body["answer"] == "The current dataset version is v2.0."
    assert body["tools_used"][0]["status"] == "ok" and body["request_id"]
