# Airflow image for this project = official Airflow + git + a separate Python
# environment that holds the training libraries (so they cannot clash with Airflow's own).
FROM apache/airflow:slim-3.4.0

# git is needed so the pipeline can read which dataset version (Git tag) it trained on.
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory '*'
USER airflow

# Training environment used by the DAG tasks (see PIPELINE_PYTHON in docker-compose.yml).
COPY requirements-train.txt /tmp/requirements-train.txt
RUN (python -m venv /home/airflow/train-venv \
        || (pip install --no-cache-dir virtualenv && python -m virtualenv /home/airflow/train-venv)) \
    && /home/airflow/train-venv/bin/pip install --no-cache-dir -r /tmp/requirements-train.txt

# Writes the Airflow login from environment variables when the container starts.
COPY set_password.py /opt/airflow/set_password.py
