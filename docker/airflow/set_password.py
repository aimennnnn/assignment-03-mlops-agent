"""Create Airflow's password file from environment variables (so no password is in Git)."""

import json
import os

user = os.environ.get("AIRFLOW_USERNAME", "admin")
password = os.environ["AIRFLOW_PASSWORD"]
with open("/opt/airflow/passwords.json", "w") as f:
    json.dump({user: password}, f)
print(f"Airflow login written for user '{user}'")
