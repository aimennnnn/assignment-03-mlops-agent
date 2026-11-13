"""The agent loop: the LLM picks tools, this code checks and runs them, the LLM answers.

    user message -> LLM -> tool calls -> (validate, check permission, run, log) -> LLM -> answer

The safety rules live here in code, not in the LLM:
  - arguments are validated against the tool's schema before it runs
  - a state-changing tool never runs unless the user confirmed that exact action
  - every tool call is logged with a request id, status and duration
  - failed tools are listed in the answer, so missing evidence is never hidden
"""

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as ToolTimeout
from datetime import datetime, timezone

from pydantic import ValidationError

from agent.llm import LLM, LLMError
from agent.registry import BY_NAME, llm_tool_specs
from agent.tracing import log_event

MAX_STEPS = 12  # most LLM rounds per request
TOOL_TIMEOUT = 150  # seconds before a tool call is given up on
MAX_RESULT_CHARS = 3000  # how much of a tool result the LLM gets to read
# A retraining assessment must look at all four systems (tool names start with these).
SYSTEMS = {"mlflow_": "MLflow", "dvc_": "DVC", "airflow_": "Airflow", "k8s_": "Kubernetes"}

SYSTEM_PROMPT = """You are an MLOps assistant for one ML project. You can inspect DVC \
(datasets), MLflow (experiments, model registry), Airflow (training pipeline) and \
Kubernetes (the deployed prediction API) through tools. Today is {today} (UTC).

Rules:
1. Every operational fact you state (versions, metrics, statuses, ids, dates) must come \
from a tool result in this conversation. Never use memory or guesses for such facts.
2. If a tool returns "ok": false, that information is unavailable. Say so, give the \
tool's error, and do not fill the gap.
3. Call every tool you need in one step when the calls do not depend on each other.
4. For multi-part questions, gather evidence from each relevant system before answering.
5. Before saying whether retraining is needed, call policy_get_retraining_policy and \
check every accepted_evidence item against tool results from all four systems: MLflow \
(best run; mlflow_get_registered_models for the champion version and its age), DVC \
(current dataset version), Airflow (airflow_get_dag_runs: did recent pipeline runs pass?) \
and Kubernetes (deployed version, health). Also compare the model version the deployment \
reports with the registry champion: if they differ, the deployment is behind and needs a \
redeploy, which is not the same as retraining.
6. The highest metric alone does not make a model production-ready; pass on the cautions.
7. Never trigger a pipeline or restart a deployment unless the user explicitly asks for \
that action. Such tools are not executed until the user confirms; when a tool result \
says needs_confirmation, ask the user to confirm and describe the action only with what \
the tool's own description says; do not describe steps you have not read from a tool.

Answer format: short and concrete. For analysis questions use three parts:
Known facts (each with the tool it came from), Uncertainties, Recommended actions."""


def _compact(value, depth=0):
    """Shorten long lists and strings so tool results stay small for the LLM."""
    if isinstance(value, dict):
        return {k: _compact(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        short = [_compact(v, depth + 1) for v in value[:8]]
        return short + [f"... {len(value) - 8} more"] if len(value) > 8 else short
    if isinstance(value, str) and len(value) > 300:
        return value[:300] + "..."
    if isinstance(value, float):
        return round(value, 4)
    return value


def _for_llm(result):
    text = json.dumps(_compact(result), separators=(",", ":"), default=str)
    return text if len(text) <= MAX_RESULT_CHARS else text[:MAX_RESULT_CHARS] + "...(cut)"


def execute_tool(name, raw_arguments, request_id, confirmed=False):
    """Validate, permission-check, run and log ONE tool call.

    Returns (result dict, record dict). Never raises.
    """
    started = time.time()
    arguments, status, result = {}, "ok", None
    tool = BY_NAME.get(name)
    try:
        if tool is None:
            status, result = "unknown_tool", {"ok": False, "error": f"No tool named '{name}'"}
        else:
            if isinstance(raw_arguments, str):
                raw_arguments = json.loads(raw_arguments or "{}")
            arguments = tool.args.model_validate(raw_arguments or {}).model_dump()
            # Empty values mean "use the tool's default".
            kwargs = {k: v for k, v in arguments.items() if v not in ("", [], {})}
            if tool.changes_state and not confirmed:
                status = "needs_confirmation"
                result = {
                    "ok": False,
                    "needs_confirmation": True,
                    "message": f"NOT executed. '{name}' changes state and needs the user's "
                    "confirmation. Tell the user what it would do and ask them to confirm.",
                }
            else:
                if tool.changes_state:
                    kwargs["confirm"] = True
                pool = ThreadPoolExecutor(max_workers=1)
                try:
                    result = pool.submit(tool.func, **kwargs).result(timeout=TOOL_TIMEOUT)
                finally:
                    pool.shutdown(wait=False)  # do not hang on a stuck tool
                if not result.get("ok"):
                    status = "error"
    except (ValidationError, json.JSONDecodeError) as e:
        detail = "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()) \
            if isinstance(e, ValidationError) else "arguments were not valid JSON"
        status, result = "invalid_arguments", {"ok": False,
                                               "error": f"Invalid arguments: {detail}"}
    except ToolTimeout:
        status, result = "timeout", {"ok": False,
                                     "error": f"Tool timed out after {TOOL_TIMEOUT}s"}
    except Exception as e:  # a tool must never crash the agent
        status, result = "error", {"ok": False, "error": f"Tool crashed ({type(e).__name__})"}

    record = {
        "tool": name,
        "arguments": arguments,
        "status": status,
        "ok": status == "ok",
        "duration_ms": int((time.time() - started) * 1000),
        "error": result.get("error") if status not in ("ok", "needs_confirmation") else None,
    }
    log_event("tool_call", request_id, **record)
    return result, record


def _missing_evidence(records):
    """For a retraining assessment: which of the four systems were never consulted."""
    used = [r["tool"] for r in records]
    if "policy_get_retraining_policy" not in used:
        return []  # not a retraining question
    missing = [label for prefix, label in SYSTEMS.items()
               if not any(name.startswith(prefix) for name in used)]
    if "mlflow_get_registered_models" not in used:
        # the policy is about the champion model, which only the registry can identify
        missing.append("the Model Registry (mlflow_get_registered_models)")
    return missing


def _describe(name, arguments):
    tool = BY_NAME[name]
    shown = {k: v for k, v in arguments.items() if v not in ("", [], {})}
    return f"{tool.description} Arguments: {shown or 'defaults'}"


def run_agent(message, confirm_action=None, llm=None, request_id=None):
    """Answer one user message. Returns a dict shaped like schemas.ChatResponse."""
    request_id = request_id or uuid.uuid4().hex[:12]
    llm = llm or LLM()
    records, pending, llm_calls = [], None, 0
    log_event("request", request_id, message=message[:200], confirming=bool(confirm_action))

    # The user confirmed a pending action: run exactly that action, nothing else.
    if confirm_action:
        tool = BY_NAME.get(confirm_action.get("tool"))
        if tool is None or not tool.changes_state:
            return _response(request_id, "That action cannot be confirmed: it is not a "
                             "known state-changing tool.", records, None, 0)
        result, record = execute_tool(tool.name, confirm_action.get("arguments", {}),
                                      request_id, confirmed=True)
        records.append(record)
        message = (f"{message}\n\n[The user confirmed. {tool.name} was executed once, result: "
                   f"{_for_llm(result)}. Do not run it again. Finish the request with "
                   "read-only tools if needed, then report what happened.]")

    today = datetime.now(timezone.utc).date().isoformat()
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(today=today)},
                {"role": "user", "content": message}]
    answer, nudged = None, False
    try:
        for _ in range(MAX_STEPS):
            reply = llm.chat(messages, llm_tool_specs())
            llm_calls += 1
            calls = reply.get("tool_calls") or []
            if not calls:
                answer = (reply.get("content") or "").strip()
                missing = _missing_evidence(records)
                if missing and not nudged:
                    # The model tried to judge retraining without checking every system.
                    nudged = True
                    log_event("evidence_check", request_id, missing=missing)
                    messages.append({"role": "assistant", "content": answer})
                    messages.append({"role": "user", "content":
                                     "[Automatic check] Answer not accepted yet: a retraining "
                                     f"assessment needs evidence from {', '.join(missing)} too. "
                                     "Call tools for those systems now, then answer again."})
                    continue
                break
            messages.append({"role": "assistant", "content": reply.get("content") or "",
                             "tool_calls": calls})
            for call in calls:
                name = call["function"]["name"]
                if confirm_action and name in BY_NAME and BY_NAME[name].changes_state:
                    # One confirmation allows one action. Refuse any further change.
                    result = {"ok": False, "error": "A state-changing action was already "
                              "executed in this request; no further changes are allowed."}
                    messages.append({"role": "tool", "tool_call_id": call.get("id", name),
                                     "content": _for_llm(result)})
                    continue
                result, record = execute_tool(name, call["function"].get("arguments"),
                                              request_id)
                records.append(record)
                if record["status"] == "needs_confirmation" and pending is None:
                    pending = {"tool": name, "arguments": record["arguments"],
                               "description": _describe(name, record["arguments"])}
                messages.append({"role": "tool", "tool_call_id": call.get("id", name),
                                 "content": _for_llm(result)})
        else:
            answer = "I stopped because the request needed too many steps. " \
                     "The tool results gathered so far are listed below."
    except LLMError as e:
        log_event("llm_error", request_id, error=str(e))
        answer = f"I could not complete this request because the language model failed: {e}"

    return _response(request_id, answer or "(the model returned an empty answer)",
                     records, pending, llm_calls)


def _response(request_id, answer, records, pending, llm_calls):
    """Build the reply and add notes the LLM cannot leave out."""
    failed = [r for r in records if r["status"] in ("error", "timeout", "invalid_arguments",
                                                    "unknown_tool")]
    if failed:
        answer += "\n\nUnavailable evidence (these tool calls failed, so nothing above " \
                  "relies on them):\n" + "\n".join(
                      f"- {r['tool']}: {r['error']}" for r in failed)
    if not records:
        answer += "\n\nNote: no tools were called, so this reply contains no verified " \
                  "operational facts."
    if pending and all(r["status"] == "needs_confirmation" for r in records):
        # Nothing has run yet, so say exactly that in fixed words (no model-written detail).
        answer = ("Nothing has been changed yet. To do this I need to run "
                  f"{pending['tool']}.\n{pending['description']}")
    if pending:
        answer += f"\n\nWaiting for your confirmation to run: {pending['tool']}."
    log_event("response", request_id, tools=len(records), failed=len(failed),
              needs_confirmation=bool(pending), llm_calls=llm_calls)
    return {
        "request_id": request_id,
        "answer": answer,
        "tools_used": records,
        "failed_tools": [r["tool"] for r in failed],
        "needs_confirmation": bool(pending),
        "pending_action": pending,
        "llm_calls": llm_calls,
    }
