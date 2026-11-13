"""Training DAG: validate -> preprocess -> train -> evaluate -> quality gate.

Each task runs one stage of training/pipeline.py.
If a task fails, Airflow does not run the tasks after it, so a failed
validation or a failed quality gate can never lead to a promoted model.
"""

import os
from datetime import timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG, Param

PROJECT = os.environ.get("PROJECT_DIR", "/opt/project")  # where the repo is mounted
PYTHON = os.environ.get("PIPELINE_PYTHON", "/opt/train-venv/bin/python")

# Settings every stage receives. Airflow fills in the {{ ... }} parts when a task runs.
ARGS = (
    "--run-dir '/tmp/pipeline_runs/{{ run_id }}' --pipeline-run '{{ run_id }}' "
    "--model {{ params.model }} --n-estimators {{ params.n_estimators }} "
    "--max-depth {{ params.max_depth }} --C {{ params.C }} --min-f1 {{ params.min_f1 }}"
)


def stage(name):
    """The shell command that runs one stage of training/pipeline.py."""
    return f"cd {PROJECT} && {PYTHON} -m training.pipeline {name} " + ARGS


with DAG(
    dag_id="training_pipeline",
    description="Validate data, train a model, log it to MLflow and apply a quality gate",
    schedule=None,  # runs only when someone (or the agent) triggers it
    catchup=False,
    is_paused_upon_creation=False,
    tags=["mlops", "training"],
    default_args={
        "retries": 2,  # retry behaviour: try a failed task 2 more times
        "retry_delay": timedelta(seconds=20),
        "execution_timeout": timedelta(minutes=10),
    },
    params={  # settings you can change when triggering; Airflow checks their type
        "model": Param("logreg", enum=["logreg", "rf"]),
        "n_estimators": Param(100, type="integer", minimum=1, maximum=1000),
        "max_depth": Param(5, type="integer", minimum=1, maximum=50),
        "C": Param(1.0, type="number", minimum=0.0001, maximum=1000),
        "min_f1": Param(0.95, type="number", minimum=0, maximum=1),
    },
) as dag:
    # A bad dataset or a failed gate will not fix itself, so these two do not retry.
    validate = BashOperator(
        task_id="validate_dataset", bash_command=stage("validate"), retries=0
    )
    preprocess = BashOperator(
        task_id="preprocess_data", bash_command=stage("preprocess")
    )
    train = BashOperator(task_id="train_model", bash_command=stage("train"))
    evaluate = BashOperator(
        task_id="evaluate_and_log_to_mlflow", bash_command=stage("evaluate")
    )
    gate = BashOperator(
        task_id="apply_quality_gate", bash_command=stage("gate"), retries=0
    )

    # Task dependencies: each stage waits for the one before it.
    validate >> preprocess >> train >> evaluate >> gate
