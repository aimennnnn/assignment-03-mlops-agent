"""Retraining policy tool: gives the agent the explicit rules from policy.yaml."""

from datetime import datetime, timezone
from pathlib import Path

import yaml

POLICY_FILE = Path("policy.yaml")


def get_retraining_policy():
    """Return the explicit retraining policy (freshness, data and quality rules)."""
    try:
        policy = yaml.safe_load(POLICY_FILE.read_text())["retraining"]
    except (OSError, KeyError, TypeError, yaml.YAMLError) as e:
        return {"ok": False, "tool": "get_retraining_policy",
                "error": f"policy.yaml could not be read ({type(e).__name__})"}
    return {
        "ok": True,
        "tool": "get_retraining_policy",
        "today": datetime.now(timezone.utc).date().isoformat(),
        "policy": policy,
    }


POLICY_TOOLS = [get_retraining_policy]
