"""Structured logs: one JSON line per event, with a request id, and no secrets."""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

LOG_FILE = Path("logs/agent.jsonl")
SECRET_WORDS = ("password", "token", "secret", "api_key", "authorization")

_logger = logging.getLogger("agent")
if not _logger.handlers:
    _logger.setLevel(logging.INFO)
    _logger.addHandler(logging.StreamHandler(sys.stderr))
    _logger.propagate = False


def redact(value):
    """Replace anything that looks like a secret with ***."""
    if isinstance(value, dict):
        return {k: "***" if any(w in k.lower() for w in SECRET_WORDS) else redact(v)
                for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


def log_event(event, request_id, **fields):
    """Write one structured log line to logs/agent.jsonl and to the console."""
    line = json.dumps(
        {"time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
         "event": event, "request_id": request_id, **redact(fields)},
        default=str,
    )
    _logger.info(line)
    try:
        LOG_FILE.parent.mkdir(exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass  # logging must never break the agent
