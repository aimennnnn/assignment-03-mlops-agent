"""Structured logging must never write secrets."""

import json

from agent.tracing import log_event, redact


def test_secret_values_are_redacted(logs_in_temp_folder):
    log_event("tool_call", "req-1", tool="x", status="ok", duration_ms=5,
              arguments={"api_key": "sk-SECRET", "params": {"password": "hunter2", "n": 3}})

    text = logs_in_temp_folder.read_text()
    assert "sk-SECRET" not in text and "hunter2" not in text
    line = json.loads(text)
    assert line["arguments"]["api_key"] == "***"
    assert line["arguments"]["params"] == {"password": "***", "n": 3}


def test_redact_keeps_ordinary_values():
    assert redact({"tool": "dvc_list", "token": "abc", "items": [{"secret_note": "x"}]}) == {
        "tool": "dvc_list", "token": "***", "items": [{"secret_note": "***"}]}
