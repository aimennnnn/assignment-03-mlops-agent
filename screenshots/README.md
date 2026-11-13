# Screenshots

Evidence for the report. Save each screenshot here with the name shown.

| File | What it shows |
|---|---|
| `01-dvc-versions.png` | `dvc diff v1.0 v2.0` and the checkout back to v1.0 (400 rows) and v2.0 (569 rows) |
| `02-dvc-tools.png` | Output of `python -m agent.tools.dvc_tools` |
| `03-mlflow-runs.png` | MLflow "Training runs" page with the runs and their metrics |
| `04-mlflow-tools.png` | The `cautions` part of `python -m agent.tools.mlflow_tools` |
| `05-airflow-dag.png` | Airflow page of `training_pipeline` with a green run and a failed run |
| `06-airflow-failed-task.png` | Log of the failed task ("Model was NOT promoted" or the validation error) |
| `07-k8s-healthy.png` | `kubectl get pods,svc -n mlops` with two pods on 1/1 Running |
| `08-k8s-crashloop.png` | Tool output while the pods were in CrashLoopBackOff |
| `09-agent-scenario.png` | The agent answering the main multi-tool question |
| `10-agent-confirmation.png` | The agent asking for confirmation before triggering the pipeline |
| `11-agent-api-docs.png` | The FastAPI page at http://127.0.0.1:8010/docs |
| `12-tests-passed.png` | `python -m pytest -v` ending in "51 passed" |
| `13-ci-passed.png` | The green GitHub Actions run on a pull request |
| `14-logs.png` | A few lines of `logs/agent.jsonl` |
