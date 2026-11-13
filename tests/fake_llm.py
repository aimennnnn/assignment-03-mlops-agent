"""A scripted stand-in for the language model, so tests need no API key or network.

Give it a list of steps. Each step is either a list of (tool_name, arguments) to call,
or a string (the final answer). It also records what it was sent.
"""

import json


class FakeLLM:
    def __init__(self, steps):
        self.steps = list(steps)
        self.seen = []  # the messages it received on each call

    def chat(self, messages, tools):
        self.seen.append(list(messages))
        step = self.steps.pop(0) if self.steps else "done"
        if isinstance(step, str):
            return {"role": "assistant", "content": step}
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": f"call_{i}", "type": "function",
                 "function": {"name": name, "arguments": json.dumps(arguments)}}
                for i, (name, arguments) in enumerate(step)
            ],
        }

    def tool_results(self):
        """Every tool result the agent sent back to the model, as dicts."""
        last = self.seen[-1] if self.seen else []
        out = []
        for m in last:
            if m["role"] == "tool":
                try:
                    out.append(json.loads(m["content"]))
                except ValueError:
                    out.append({"raw": m["content"]})
        return out
