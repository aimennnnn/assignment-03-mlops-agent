"""Task 3 - Apache Airflow workflow tools.

Each function below is one "tool" the AI agent can call.
They talk to the Airflow 3 REST API with a login token (authenticated requests).
Only one tool changes anything (trigger_training_dag) and it needs confirm=True.

Try it by hand:   python -m agent.tools.airflow_tools
Trigger a run:    python -m agent.tools.airflow_tools --trigger
Trigger a run that fails the quality gate on purpose (weak model):
                  python -m agent.tools.airflow_tools --trigger-weak
"""

import json
import os
import re
import sys
import time
from urllib.parse import quote

import requests
from dotenv import load_dotenv

load_dotenv()  # reads AIRFLOW_PASSWORD etc. from the .env file (never saved in Git)

AIRFLOW_URL = os.environ.get("AIRFLOW_URL", "http://127.0.0.1:8080").rstrip("/")
USERNAME = os.environ.get("AIRFLOW_USERNAME", "admin")
PASSWORD = os.environ.get("AIRFLOW_PASSWORD", "")

# Safety: the ONLY DAGs the agent is allowed to trigger.
APPROVED_DAGS = ["training_pipeline"]
TIMEOUT = 15  # seconds per request
SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
FINISHED = {"success", "failed"}

_token = None


def _login():
    """Get a login token from Airflow (POST /auth/token)."""
    global _token
    r = requests.post(
        f"{AIRFLOW_URL}/auth/token",
        json={"username": USERNAME, "password": PASSWORD},
        timeout=TIMEOUT,
    )
    if r.status_code >= 400:
        raise RuntimeError("Airflow login failed: check AIRFLOW_USERNAME / AIRFLOW_PASSWORD")
    _token = r.json()["access_token"]


def _api(method, path, **kwargs):
    """Send one authenticated request to the Airflow REST API and return the JSON."""
    if _token is None:
        _login()
    for attempt in (1, 2):
        r = requests.request(
            method,
            f"{AIRFLOW_URL}/api/v2{path}",
            headers={"Authorization": f"Bearer {_token}", "Accept": "application/json"},
            timeout=TIMEOUT,
            **kwargs,
        )
        if r.status_code == 401 and attempt == 1:  # token expired -> log in again once
            _login()
            continue
        break
    if r.status_code == 404:
        raise ValueError("Not found in Airflow: " + str(r.json().get("detail", path))[:200])
    if r.status_code >= 400:
        raise RuntimeError(f"Airflow answered {r.status_code}: {r.text[:300]}")
    return r.json()


def _safe(tool):
    """Turn any failure into {"ok": False, "error": ...} so the agent can report it."""

    def wrapper(*args, **kwargs):
        try:
            return {"ok": True, "tool": tool.__name__, **tool(*args, **kwargs)}
        except requests.exceptions.Timeout:
            error = f"Airflow at {AIRFLOW_URL} did not answer within {TIMEOUT}s"
        except requests.exceptions.RequestException:
            error = f"Airflow at {AIRFLOW_URL} is not reachable"
        except (ValueError, RuntimeError, PermissionError) as e:
            error = str(e)
        return {"ok": False, "tool": tool.__name__, "error": error}

    wrapper.__name__ = tool.__name__
    wrapper.__doc__ = tool.__doc__
    return wrapper


def _check_id(value, what):
    if not SAFE_ID.match(value or ""):
        raise ValueError(f"'{value}' is not a valid {what}")
    return value


def _run_path(dag_id, run_id):
    return f"/dags/{_check_id(dag_id, 'DAG id')}/dagRuns/{quote(run_id, safe='')}"


def _run_summary(run):
    return {
        "run_id": run["dag_run_id"],
        "state": run["state"],
        "started": run.get("start_date"),
        "ended": run.get("end_date"),
        "duration_seconds": run.get("duration") and round(run["duration"], 1),
        "triggered_by": run.get("triggered_by"),
        "params": run.get("conf") or {},
    }


def _latest_run_id(dag_id):
    runs = _api("GET", f"/dags/{_check_id(dag_id, 'DAG id')}/dagRuns",
                params={"order_by": "-run_after", "limit": 1})["dag_runs"]
    if not runs:
        raise ValueError(f"DAG '{dag_id}' has no runs yet")
    return runs[0]["dag_run_id"]


def _tasks(dag_id, run_id):
    """Task states of one run, in the order they ran."""
    items = _api("GET", _run_path(dag_id, run_id) + "/taskInstances")["task_instances"]
    items.sort(key=lambda t: t.get("start_date") or "9999")
    return [
        {
            "task_id": t["task_id"],
            "state": t["state"],
            "tries_used": t["try_number"],
            "retries_allowed": t["max_tries"],
            "started": t.get("start_date"),
            "duration_seconds": t.get("duration") and round(t["duration"], 1),
        }
        for t in items
    ]


@_safe
def list_dags():
    """List the DAGs Airflow knows about, and any DAG files that failed to load."""
    dags = _api("GET", "/dags")["dags"]
    errors = _api("GET", "/importErrors")["import_errors"]
    return {
        "count": len(dags),
        "dags": [
            {
                "dag_id": d["dag_id"],
                "description": d.get("description"),
                "is_paused": d["is_paused"],
                "schedule": d.get("timetable_summary"),
                "tags": [t["name"] for t in d.get("tags", [])],
                "agent_may_trigger": d["dag_id"] in APPROVED_DAGS,
            }
            for d in dags
        ],
        "import_errors": [
            {"file": e.get("filename"), "error": str(e.get("stack_trace"))[-300:]} for e in errors
        ],
    }


@_safe
def get_dag_runs(dag_id="training_pipeline", limit=5):
    """List the most recent runs of a DAG (newest first) with their state."""
    runs = _api("GET", f"/dags/{_check_id(dag_id, 'DAG id')}/dagRuns",
                params={"order_by": "-run_after", "limit": int(limit)})["dag_runs"]
    return {"dag_id": dag_id, "count": len(runs), "runs": [_run_summary(r) for r in runs]}


@_safe
def get_dag_run_status(dag_id="training_pipeline", run_id=None, wait_seconds=0):
    """Get one run's state and the state of each task. Uses the latest run if none is given.

    wait_seconds > 0 waits (up to 180s) for a running run to finish first.
    """
    run_id = run_id or _latest_run_id(dag_id)
    deadline = time.time() + min(int(wait_seconds), 180)
    while True:
        run = _api("GET", _run_path(dag_id, run_id))
        if run["state"] in FINISHED or time.time() >= deadline:
            break
        time.sleep(3)
    return {
        "dag_id": dag_id,
        **_run_summary(run),
        "finished": run["state"] in FINISHED,
        "tasks": _tasks(dag_id, run_id),
    }


@_safe
def trigger_training_dag(dag_id="training_pipeline", params=None, confirm=False):
    """Start a run of an approved training DAG. Changes state, so it needs confirm=True."""
    if dag_id not in APPROVED_DAGS:
        raise PermissionError(
            f"DAG '{dag_id}' is not approved for triggering. Approved: {APPROVED_DAGS}"
        )
    params = params or {}
    if not confirm:
        return {
            "triggered": False,
            "needs_confirmation": True,
            "message": f"This would start DAG '{dag_id}' with params {params}. "
            "Ask the user to confirm, then call again with confirm=True.",
        }
    run = _api("POST", f"/dags/{dag_id}/dagRuns", json={"logical_date": None, "conf": params})
    return {"triggered": True, "dag_id": dag_id, **_run_summary(run)}


@_safe
def get_failed_tasks(dag_id="training_pipeline", run_id=None):
    """Find which tasks failed in a run, and which were skipped because of that."""
    run_id = run_id or _latest_run_id(dag_id)
    tasks = _tasks(dag_id, run_id)
    return {
        "dag_id": dag_id,
        "run_id": run_id,
        "run_state": _api("GET", _run_path(dag_id, run_id))["state"],
        "failed_tasks": [t for t in tasks if t["state"] == "failed"],
        "not_run_because_of_failure": [
            t["task_id"] for t in tasks if t["state"] == "upstream_failed"
        ],
    }


@_safe
def get_task_diagnostics(task_id, dag_id="training_pipeline", run_id=None, log_lines=25):
    """Get one task's execution details and the last lines of its log (for diagnosing)."""
    run_id = run_id or _latest_run_id(dag_id)
    path = _run_path(dag_id, run_id) + f"/taskInstances/{_check_id(task_id, 'task id')}"
    t = _api("GET", path)
    lines = []
    if t["try_number"]:  # a task that never started has no log
        log = _api("GET", f"{path}/logs/{t['try_number']}")
        for entry in log.get("content", []):
            text = str(entry.get("event", ""))
            if text.startswith("::") or "/logs/dag_id=" in text:
                continue  # layout markers, not real log lines
            for err in entry.get("error_detail") or []:
                text += f" [{err.get('exc_type')}: {err.get('exc_value')}]"
            lines.append(f"{entry.get('level', 'info').upper()}: {text[:300]}")
    return {
        "dag_id": dag_id,
        "run_id": run_id,
        "task_id": task_id,
        "state": t["state"],
        "operator": t.get("operator_name"),
        "tries_used": t["try_number"],
        "retries_allowed": t["max_tries"],
        "started": t.get("start_date"),
        "ended": t.get("end_date"),
        "duration_seconds": t.get("duration") and round(t["duration"], 1),
        "error_lines": [x for x in lines if "ERROR" in x or "FAILED" in x][-5:],
        "log_tail": lines[-int(log_lines):],
    }


# The list the agent will pick from in Step 5.
AIRFLOW_TOOLS = [
    list_dags,
    get_dag_runs,
    get_dag_run_status,
    trigger_training_dag,
    get_failed_tasks,
    get_task_diagnostics,
]


def _show(result):
    print(json.dumps(result, indent=2), "\n")


if __name__ == "__main__":
    if "--trigger" in sys.argv or "--trigger-weak" in sys.argv:
        weak = {"model": "rf", "n_estimators": 10, "max_depth": 2}
        started = trigger_training_dag(
            params=weak if "--trigger-weak" in sys.argv else None, confirm=True
        )
        _show(started)
        if started["ok"]:
            print("waiting for the run to finish...\n")
            _show(get_dag_run_status(run_id=started["run_id"], wait_seconds=180))
    else:
        _show(list_dags())
        _show(get_dag_runs(limit=3))
        _show(get_dag_run_status())
        failed = get_failed_tasks()
        _show(failed)
        if failed["ok"] and failed["failed_tasks"]:
            _show(get_task_diagnostics(failed["failed_tasks"][0]["task_id"]))
        _show(trigger_training_dag())  # no confirm -> asks first, starts nothing
        _show(trigger_training_dag("some_other_dag", confirm=True))  # not approved -> refused
        _show(get_dag_runs("no_such_dag"))  # does not exist -> clean error, no guess
