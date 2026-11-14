# Screenshots

Evidence used in [REPORT.md](../REPORT.md). Text output from the tools is in [`evidence/`](../evidence).

| File | What it shows |
|---|---|
| `03-mlflow-training-runs.png` | MLflow: training runs of the `cancer-logreg` experiment |
| `05-airflow-dashboard.png` | Airflow: DAG runs and task instances, including the failed ones |
| `06-airflow-failed-runs.png` | Airflow: the two deliberately failed runs of `training_pipeline` |
| `07-k8s-deployment-and-pods.png` | Kubernetes (namespace `mlops`): `cancer-api` Available 2/2, both pods Running |
| `08-k8s-service-and-configmap.png` | Kubernetes: the ClusterIP Service and the ConfigMap |
| `09-agent-scenario-and-pods.png` | The agent answering the main multi-tool question, with the tools it used |
| `11-agent-api-docs.png` | The agent's FastAPI interface at `/docs` |
| `13-ci-passed-on-pull-request.png` | GitHub Actions: lint and tests passing on a pull request |
| `14-ci-actions-run.png` | GitHub Actions: the run on `main` |
| `15-pull-request-merged.png` | The pull request merged after its check passed |
