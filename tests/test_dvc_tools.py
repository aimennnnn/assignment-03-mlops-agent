"""DVC tools with the command line mocked: only approved commands, safe input, timeouts."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from agent.tools import dvc_tools as dvc


@pytest.fixture
def shell(monkeypatch):
    """A fake command line. Returns the list of commands the tools tried to run."""
    ran = []
    answers = {
        ("dvc", "list"): json.dumps([{"path": "data\\cancer.csv", "md5": "abc", "size": 10}]),
        ("dvc", "status"): "{}",
        ("dvc", "remote"): "localremote  ../dvc-remote (default)",
        ("git", "describe"): "v2.0",
        ("git", "rev-parse"): "c986c4e",
    }

    def run(cmd, **kwargs):
        ran.append(cmd)
        return SimpleNamespace(returncode=0, stdout=answers.get(tuple(cmd[:2]), ""), stderr="")

    monkeypatch.setattr(dvc.subprocess, "run", run)
    return ran


def test_lists_tracked_datasets(shell):
    result = dvc.list_tracked_datasets()
    assert result["ok"] is True
    assert result["datasets"] == [{"path": "data/cancer.csv", "md5": "abc", "size_bytes": 10}]


def test_current_version_reports_tag_and_commit(shell):
    result = dvc.get_current_dataset_version()
    assert result["version"] == "v2.0" and result["git_commit"] == "c986c4e"


def test_only_approved_commands_can_run(shell):
    """dvc push, git reset and friends are not on the approved list."""
    for cmd in (["dvc", "push"], ["git", "reset", "--hard"], ["rm", "-rf", "data"]):
        with pytest.raises(PermissionError):
            dvc._run(cmd)
    assert shell == []


def test_unsafe_version_name_is_rejected(shell):
    result = dvc.compare_dataset_versions("--force", "v1.0; rm -rf .")
    assert result["ok"] is False and "not a valid version" in result["error"]
    assert shell == []  # nothing was executed


def test_dvc_timeout_is_reported(monkeypatch):
    def hang(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 60)

    monkeypatch.setattr(dvc.subprocess, "run", hang)
    result = dvc.check_data_changed()
    assert result["ok"] is False and "timed out" in result["error"]


def test_dvc_not_installed(monkeypatch):
    def missing(cmd, **kwargs):
        raise FileNotFoundError("dvc")

    monkeypatch.setattr(dvc.subprocess, "run", missing)
    assert dvc.list_tracked_datasets()["error"] == "dvc or git is not installed"
