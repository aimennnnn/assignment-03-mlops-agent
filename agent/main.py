"""FastAPI interface of the agent: validates the request, runs the agent, returns the reply.

    uvicorn agent.main:app --port 8010
    then open http://127.0.0.1:8010/docs
"""

from fastapi import FastAPI

from agent.orchestrator import run_agent
from agent.registry import describe_tools
from agent.schemas import ChatRequest, ChatResponse

app = FastAPI(
    title="MLOps AI Agent",
    description="Ask about datasets (DVC), experiments (MLflow), pipelines (Airflow) "
    "and the deployment (Kubernetes) in plain English.",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tools")
def tools():
    """Every tool the agent can use and its permission level."""
    return {"count": len(describe_tools()), "tools": describe_tools()}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Send a message. To confirm a pending action, send it back in confirm_action."""
    action = request.confirm_action.model_dump() if request.confirm_action else None
    return run_agent(request.message, confirm_action=action)
