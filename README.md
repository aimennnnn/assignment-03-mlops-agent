# MLOps AI Agent

Assignment 3, MLOps & Production AI Engineering, Fall 2026.

An AI agent that answers plain-English questions about an ML project by calling real
tools. It can inspect datasets (DVC), experiments and models (MLflow), the training
pipeline (Airflow) and the deployed prediction API (Kubernetes). Every operational
fact in an answer comes from a tool result, never from the model's memory.

```
python -m agent.ask "Find the best recent model, identify the dataset version used to
train it, check the deployment status, and explain whether retraining is necessary."
```

See [architecture.md](architecture.md) for the design and the full tool list.

## What is in the repository

```
agent/                  the AI agent
  main.py               FastAPI interface (/chat, /tools, /health)
  orchestrator.py       agent loop: LLM picks tools, code checks and runs them
  registry.py           all 25 tools with their permissions
  schemas.py            structured inputs and outputs
  llm.py                LLM client (any chat-completions provider)
  tracing.py            structured JSON logs
  ask.py                ask questions from the terminal
  tools/                dvc_tools, mlflow_tools, airflow_tools, kubernetes_tools, policy_tools
api/main.py             FastAPI prediction service that runs on Kubernetes
dags/training_pipeline.py   Airflow DAG: validate -> preprocess -> train -> evaluate -> gate
training/               train.py, pipeline.py (the DAG stages), export_model.py
kubernetes/             namespace, configmap, deployment, service (+ failure-demo/)
docker/airflow/         Airflow image used by docker-compose.yml
tests/                  52 automated tests (no real services needed)
data/                   dataset tracked by DVC (pointer file only; data is in dvc-storage/)
dvc-storage/            the DVC remote (see Known limitations)
policy.yaml             explicit rules for recommending retraining
.github/workflows/ci.yml    lint + tests on every pull request
```

## Requirements

- Windows 10/11 with PowerShell (commands below), or Linux/macOS with small path changes
- Python 3.13 and Git
- Docker Desktop with Kubernetes enabled (Settings -> Kubernetes)
- A free LLM API key (the default settings use Groq: console.groq.com)

## 1. Install

```powershell
git clone https://github.com/aimennnnn/assignment-03-mlops-agent.git
cd assignment-03-mlops-agent
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
```

## 2. Configure

Copy the template and fill it in. `.env` is ignored by Git, so no secret is committed.

```powershell
Copy-Item .env.example .env
```

| Setting | Meaning |
|---|---|
| `AIRFLOW_URL`, `AIRFLOW_USERNAME`, `AIRFLOW_PASSWORD` | Airflow login. Choose any password; docker-compose creates the user with it. |
| `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` | The language model. Any provider with the chat-completions API works. |

Optional: `MLFLOW_TRACKING_URI` (default `http://127.0.0.1:5000`), `K8S_NAMESPACE` (default `mlops`).

## 3. Get the data

```powershell
dvc pull
```

This restores `data/cancer.csv` (dataset version v2.0, 569 rows). To restore an older
version, ask the agent "How do I reproduce dataset version v1.0?" or run:

```powershell
git checkout v1.0 -- data/cancer.csv.dvc
dvc pull
```

## 4. Start the services

**MLflow** (keep this terminal open):

```powershell
mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 0.0.0.0 --port 5000 --allowed-hosts "localhost:*,127.0.0.1:*,host.docker.internal:*" --workers 1
```

**Airflow** (one container; UI at http://127.0.0.1:8080, login from `.env`):

```powershell
docker compose up -d --build
```

Use `docker compose stop` / `start` to pause and resume. `docker compose down` deletes
the run history.

**First training runs** (creates experiments, a registered model and a champion):

```powershell
python -m training.train --model rf --n-estimators 200 --max-depth 8
python -m training.train --model logreg --C 1.0
python -m agent.tools.airflow_tools --trigger
```

**Prediction API on Kubernetes:**

```powershell
python -m training.export_model
docker build -t cancer-api:1.0.0 .
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/
kubectl rollout status deployment/cancer-api -n mlops
```

`export_model` copies the champion model from the MLflow registry into `api/model/`,
so the image always contains a model with known provenance.

## 5. Talk to the agent

From the terminal:

```powershell
python -m agent.ask "Which dataset version is current?"
python -m agent.ask "Compare the last five training runs and recommend the best candidate based on F1 score."
python -m agent.ask "Trigger the model training pipeline and report whether execution succeeded."
python -m agent.ask "Why is my prediction API unavailable?"
python -m agent.ask
```

The last form starts a chat that keeps asking until you type `exit`.

Through the FastAPI interface:

```powershell
uvicorn agent.main:app --port 8010
```

Open http://127.0.0.1:8010/docs. `POST /chat` takes `{"message": "..."}`. If the reply
has `needs_confirmation: true`, send the same message again with the returned
`pending_action` in `confirm_action` to approve it. `GET /tools` lists every tool and
its permission.

Each tool module can also be run alone, which is useful for checking one system:

```powershell
python -m agent.tools.dvc_tools
python -m agent.tools.mlflow_tools
python -m agent.tools.airflow_tools
python -m agent.tools.kubernetes_tools
```

## 6. Tests and lint

```powershell
python -m pytest -v
ruff check .
```

The 52 tests need no MLflow, Airflow, Kubernetes, network or API key. They use a fake
cluster (`tests/fake_k8s.py`), a scripted model (`tests/fake_llm.py`) and mocked HTTP and
command-line calls. GitHub Actions runs the same two commands on every pull request.

## Safety controls

| Control | Where |
|---|---|
| Only approved DVC/Git commands can run, all read-only | `ALLOWED` in `agent/tools/dvc_tools.py` |
| Only approved DAGs can be triggered | `APPROVED_DAGS` in `agent/tools/airflow_tools.py` |
| Kubernetes tools work in one namespace only | `_ns()` in `agent/tools/kubernetes_tools.py` |
| State-changing tools never run without the user's confirmation | `execute_tool()` in `agent/orchestrator.py` |
| One confirmation allows exactly one change | `run_agent()` in `agent/orchestrator.py` |
| Tool arguments are validated before a tool runs | `agent/schemas.py` |
| Failed tools are always listed in the answer | `_response()` in `agent/orchestrator.py` |
| A retraining assessment must consult every system | `_missing_evidence()` in `agent/orchestrator.py` |
| Logs carry a request id and never contain secrets | `agent/tracing.py` |
| A weak model or changed data can never be promoted | `validate()` and `gate()` in `training/pipeline.py` |

## Reproducing the two injected failures

**Data changed without DVC** (pipeline stops at validation):

```powershell
Add-Content data\cancer.csv "1,2,3"
python -m agent.tools.airflow_tools --trigger
python -m agent.ask "The latest training pipeline run failed. Find out which task failed and why."
dvc checkout --force
```

**Bad configuration** (pods crash-loop, API unavailable):

```powershell
kubectl apply -f kubernetes/failure-demo/configmap-broken.yaml
kubectl delete pods --all -n mlops
python -m agent.ask "Why is my prediction API unavailable?"
kubectl apply -f kubernetes/configmap.yaml
python -m agent.ask "Restart the prediction API deployment."
```

## Known limitations

- **DVC remote is a folder in this repository** (`dvc-storage/`), so anyone who clones
  can run `dvc pull`. A real project would use cloud storage (S3, GCS, Azure).
- **MLflow uses SQLite and one worker.** Fine for one user; a team would use Postgres.
- **Airflow runs in `standalone` mode** in a single container with its history inside
  the container. Production would use a separate database and executor.
- **Deploying a new champion is a manual step** (export, build, apply). The agent
  detects when the deployment is behind the registry but does not redeploy.
- **Free LLM rate limits** make multi-tool answers take up to a minute.
- **The agent is stateless between questions**; each request is answered on its own.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `kubectl` prints `invalid character '<'` | No cluster is selected. Enable Kubernetes in Docker Desktop. |
| A pipeline stage hangs on MLflow for minutes | Start MLflow with `--workers 1` (see step 4). |
| Agent says the LLM is not configured | Set the three `LLM_` values in `.env`. |
| Airflow tools say "login failed" | `AIRFLOW_PASSWORD` in `.env` differs from the running container; run `docker compose up -d`. |
