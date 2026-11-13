"""Ask the agent from the terminal.

    python -m agent.ask "Which dataset version is current?"
    python -m agent.ask            (no question = keep asking until you type exit)
"""

import logging
import sys

from agent.orchestrator import run_agent


def show(reply):
    print("\n" + reply["answer"] + "\n")
    if reply["tools_used"]:
        print("Tools used:")
        for t in reply["tools_used"]:
            print(f"  {t['status']:<18} {t['duration_ms']:>6} ms  {t['tool']}")
    print(f"(request {reply['request_id']}, {reply['llm_calls']} LLM call(s))\n")


def ask(message):
    reply = run_agent(message)
    show(reply)
    if reply["needs_confirmation"]:
        action = reply["pending_action"]
        print(f"The agent wants to run: {action['tool']}  {action['arguments']}")
        if input("Type yes to confirm, anything else to cancel: ").strip().lower() == "yes":
            show(run_agent(message, confirm_action=action))
        else:
            print("Cancelled. Nothing was changed.\n")


if __name__ == "__main__":
    logging.getLogger("agent").setLevel(logging.WARNING)  # keep the terminal readable
    if len(sys.argv) > 1:
        ask(" ".join(sys.argv[1:]))
    else:
        print("MLOps agent. Type a question, or exit to quit.")
        while True:
            question = input("you> ").strip()
            if question.lower() in ("exit", "quit"):
                break
            if question:
                ask(question)
