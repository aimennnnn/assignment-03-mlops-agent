"""Task 1 - DVC data management tools.

Each function below is one "tool" the AI agent can call.
Every tool is read-only: it runs an approved dvc/git command and returns
what the command said as a dict. Nothing is made up.

Try it by hand:   python -m agent.tools.dvc_tools
"""

import json
import os
import re
import subprocess

# Folder of the Git + DVC project. Defaults to where you run the agent from.
REPO_DIR = os.environ.get("MLOPS_REPO_DIR", ".")

# Safety: the ONLY commands these tools are allowed to run (all read-only).
ALLOWED = [
    ("dvc", "list"),
    ("dvc", "status"),
    ("dvc", "diff"),
    ("dvc", "remote", "list"),
    ("git", "rev-parse"),
    ("git", "describe"),
]

# A version name must look like v1.0, main or a4c92e1 (stops odd input like "--force").
SAFE_REV = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")


def _run(cmd):
    """Run one approved command and return its text output."""
    if not any(tuple(cmd[: len(a)]) == a for a in ALLOWED):
        raise PermissionError("Command not approved: " + " ".join(cmd[:3]))
    done = subprocess.run(cmd, cwd=REPO_DIR, capture_output=True, text=True, timeout=60)
    if done.returncode != 0:
        raise RuntimeError(done.stderr.strip() or "command failed: " + " ".join(cmd))
    return done.stdout.strip()


def _safe(tool):
    """Turn any failure into {"ok": False, "error": ...} so the agent can report it."""

    def wrapper(*args, **kwargs):
        try:
            return {"ok": True, "tool": tool.__name__, **tool(*args, **kwargs)}
        except FileNotFoundError:
            return {"ok": False, "tool": tool.__name__, "error": "dvc or git is not installed"}
        except subprocess.TimeoutExpired:
            return {"ok": False, "tool": tool.__name__, "error": "command timed out after 60s"}
        except (PermissionError, RuntimeError, ValueError) as e:
            return {"ok": False, "tool": tool.__name__, "error": str(e)}

    wrapper.__name__ = tool.__name__
    wrapper.__doc__ = tool.__doc__
    return wrapper


def _check_rev(rev):
    """Make sure a version name is safe and really exists in Git."""
    if not SAFE_REV.match(rev or ""):
        raise ValueError(f"'{rev}' is not a valid version name")
    try:
        return _run(["git", "rev-parse", "--short", rev + "^{commit}"])
    except RuntimeError:
        raise ValueError(f"Version '{rev}' does not exist in this repository") from None


def _datasets_at(rev=None):
    """List DVC-tracked files (path, md5, size), now or at an older version."""
    cmd = ["dvc", "list", ".", "-R", "--dvc-only", "--json"]
    if rev:
        cmd += ["--rev", rev]
    return [
        {"path": d["path"].replace("\\", "/"), "md5": d.get("md5"), "size_bytes": d.get("size")}
        for d in json.loads(_run(cmd))
    ]


@_safe
def list_tracked_datasets():
    """Identify the datasets tracked by DVC."""
    datasets = _datasets_at()
    return {"count": len(datasets), "datasets": datasets}


@_safe
def get_current_dataset_version():
    """Get the current dataset version (Git tag), Git commit, data hash and DVC remote."""
    version = _run(["git", "describe", "--tags", "--abbrev=0"])
    return {
        "version": version,
        # the commit where this dataset version was saved (matches MLflow's dataset tag)
        "git_commit": _run(["git", "rev-parse", "--short", version + "^{commit}"]),
        # the newest commit of the repository (code may have changed since)
        "latest_code_commit": _run(["git", "rev-parse", "--short", "HEAD"]),
        "datasets": _datasets_at(),
        "dvc_remote": _run(["dvc", "remote", "list"]),
    }


@_safe
def compare_dataset_versions(version_a, version_b):
    """Compare two dataset versions, for example 'v1.0' and 'v2.0'."""
    commit_a, commit_b = _check_rev(version_a), _check_rev(version_b)
    old = {d["path"]: d for d in _datasets_at(version_a)}
    new = {d["path"]: d for d in _datasets_at(version_b)}
    changes = []
    for path in sorted(set(old) | set(new)):
        a, b = old.get(path), new.get(path)
        changes.append(
            {
                "path": path,
                "status": "added" if not a else "deleted" if not b
                else "same" if a["md5"] == b["md5"] else "modified",
                "md5_a": a and a["md5"],
                "md5_b": b and b["md5"],
                "size_change_bytes": (b["size_bytes"] - a["size_bytes"]) if a and b else None,
            }
        )
    return {
        "version_a": {"name": version_a, "git_commit": commit_a},
        "version_b": {"name": version_b, "git_commit": commit_b},
        "changes": changes,
        "dvc_diff": json.loads(_run(["dvc", "diff", version_a, version_b, "--json"]) or "{}"),
    }


@_safe
def check_data_changed():
    """Detect whether tracked data was changed since it was last saved with DVC."""
    status = json.loads(_run(["dvc", "status", "--json"]) or "{}")
    return {"changed": bool(status), "details": status}


@_safe
def explain_reproduce_version(version):
    """Explain how to get back a previous dataset version, for example 'v1.0'."""
    commit = _check_rev(version)
    datasets = _datasets_at(version)
    pointers = " ".join(d["path"] + ".dvc" for d in datasets)
    return {
        "version": version,
        "git_commit": commit,
        "datasets": datasets,
        "steps": [
            f"git checkout {version} -- {pointers}   (restore that version's pointer file)",
            "dvc pull   (fetch that exact data from the DVC remote into data/)",
            f"To return: git checkout HEAD -- {pointers}, then dvc pull",
        ],
        "how_to_verify": "The md5 of the restored file must equal the md5 listed above.",
    }


# The list the agent will pick from in Step 5.
DVC_TOOLS = [
    list_tracked_datasets,
    get_current_dataset_version,
    compare_dataset_versions,
    check_data_changed,
    explain_reproduce_version,
]


if __name__ == "__main__":
    for result in (
        list_tracked_datasets(),
        get_current_dataset_version(),
        compare_dataset_versions("v1.0", "v2.0"),
        check_data_changed(),
        explain_reproduce_version("v1.0"),
        explain_reproduce_version("v9.9"),  # does not exist -> clean error, no guess
    ):
        print(json.dumps(result, indent=2), "\n")
