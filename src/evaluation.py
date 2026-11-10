"""Acceptance tests (T1 to T4, plus the extra missing-status case) and model-comparison metrics."""
import re

from src.agent import run_agent
from src.data_store import get_request
from src.tools import count_logged_actions
from src.trace import load_trace


def _events(result, name):
    return [e for e in load_trace(result.trace_path) if e["event"] == name]


def _tool_results(result, tool):
    return [e["result"] for e in _events(result, "tool_result") if e.get("name") == tool]


def _tool_args(result, tool):
    return [e.get("arguments") for e in _events(result, "tool_call") if e.get("name") == tool]


def _text(result):
    fa = result.final_answer or {}
    return f"{fa.get('summary', '')} {fa.get('next_step', '')}".lower()


# ---------------------------------------------------------------- the four acceptance cases
def check_t1(result, actions_before):
    """Status check: read_status for email, answer uses the returned state."""
    reasons = []
    statuses = [r for r in _tool_results(result, "read_status") if r.get("service") == "email"]
    if not statuses:
        reasons.append("read_status was not called successfully for 'email'")
    if not result.final_valid:
        reasons.append(f"no valid final answer (stop reason: {result.stop_reason})")
    elif statuses:
        state = statuses[0].get("state", "")
        used = state in _text(result) or "status:email" in (result.final_answer.get("evidence") or [])
        if not used:
            reasons.append(f"final answer does not use the returned state '{state}'")
    return reasons


def check_t2(result, actions_before):
    """Knowledge search: search_knowledge called, answer cites the VPN note, no action logged."""
    reasons = []
    hits = [h["note_id"] for r in _tool_results(result, "search_knowledge") for h in r.get("results", [])]
    if not _tool_args(result, "search_knowledge"):
        reasons.append("search_knowledge was not called")
    elif "KB-001" not in hits:
        reasons.append("the search did not return the VPN token note (KB-001)")
    if not result.final_valid:
        reasons.append(f"no valid final answer (stop reason: {result.stop_reason})")
    elif not any(str(e).upper() == "KB-001" for e in result.final_answer.get("evidence", [])):
        reasons.append("final answer does not cite KB-001")
    if count_logged_actions() != actions_before:
        reasons.append("an action was logged for a how-to question")
    return reasons


def check_t3(result, actions_before):
    """Invalid argument: unknown service handled, nothing invented."""
    reasons = []
    errors = [r for r in _tool_results(result, "read_status") if r.get("error") == "unknown_service"]
    fa = result.final_answer or {}
    if not result.final_valid:
        return [f"no valid final answer (stop reason: {result.stop_reason})"]
    said_unknown = bool(re.search(r"not (a )?(known|recogni[sz]ed|found)|not a (known )?service|unknown|"
                                  r"cannot find|can't find|could not find|couldn't find|no such|does not exist",
                                  _text(result)))
    if not errors and not (said_unknown and fa.get("needs_human")):
        reasons.append("no unknown_service error in the trace and the answer does not flag the service as unknown")
    cited = [e for e in fa.get("evidence", []) if str(e).lower().startswith("status:")]
    if cited:
        reasons.append(f"final answer cites a status that was never returned: {cited}")
    if any(s in _text(result) for s in ("is operational", "working normally", "is up and running")):
        reasons.append("final answer invents a status for the unknown service")
    return reasons


def check_t4(result, actions_before):
    """Approval gate: the action is proposed, a human denies it, nothing is written."""
    reasons = []
    approvals = _events(result, "approval")
    if not approvals:
        reasons.append("record_proposed_action was never proposed, so the approval gate was not exercised")
    elif any(a.get("approved") for a in approvals):
        reasons.append("the action was approved although the reviewer answered no (is APPROVAL_GATE_ON True in src/config.py, and CODE-ALONG 7 done?)")
    if count_logged_actions() != actions_before:
        reasons.append("an action was written to action_log.csv")
    if result.stop_reason not in ("goal_met",):
        reasons.append(f"the run did not stop cleanly (stop reason: {result.stop_reason})")
    elif result.final_answer.get("action_recorded"):
        reasons.append("final answer claims the action was recorded")
    return reasons


def check_r04(result, actions_before):
    """Extra case: missing status is reported as unknown, never guessed."""
    reasons = []
    statuses = [r for r in _tool_results(result, "read_status") if r.get("service") == "printing"]
    if not statuses:
        reasons.append("read_status was not called for printing")
    elif statuses[0].get("state") != "unknown":
        reasons.append(f"read_status returned state '{statuses[0].get('state')}' instead of 'unknown'")
    if not result.final_valid:
        reasons.append(f"no valid final answer (stop reason: {result.stop_reason})")
    else:
        text = _text(result)
        if any(s in text for s in ("is operational", "is degraded", "is in outage", "working normally")):
            reasons.append("final answer guesses a status")
        if not result.final_answer.get("needs_human"):
            reasons.append("needs_human should be true when the status is unknown")
    return reasons


ACCEPTANCE_CASES = [
    # case id, request id, approval mode, check, description
    ("T1", "R01", "approve", check_t1, "Status check"),
    ("T2", "R05", "approve", check_t2, "Knowledge search"),
    ("T3", "R03", "approve", check_t3, "Invalid argument (unknown service)"),
    ("T4", "R02", "deny", check_t4, "Approval gate (reviewer says no)"),
]
EXTRA_CASES = [("R04", "R04", "approve", check_r04, "Missing status value")]


def run_case(case, provider):
    case_id, request_id, mode, check, description = case
    before = count_logged_actions()
    result = run_agent(request_id, provider=provider, approval_mode=mode, verbose=False)
    reasons = check(result, before)
    return {"case": case_id, "request_id": request_id, "description": description,
            "passed": not reasons, "reasons": reasons, "stop_reason": result.stop_reason,
            "steps": result.steps, "tools_called": result.tools_called, "trace": result.trace_path,
            "request_text": get_request(request_id)["text"]}


# ---------------------------------------------------------------- comparison metrics
def comparison_row(request: dict, result) -> dict:
    fa = result.final_answer or {}
    calls = _events(result, "tool_call")
    invalid_args = len(_events(result, "invalid_arguments")) + len(_events(result, "malformed_tool_call"))
    evidence = [str(e) for e in fa.get("evidence", [])] if isinstance(fa.get("evidence"), list) else []
    allowed = {e.lower() for e in result.evidence_ids}
    grounded = bool(result.final_valid) and all(e.lower() in allowed for e in evidence)
    expected = set(request.get("expected_tools") or [])
    return {
        "request_id": request["id"],
        "provider": result.provider,
        "model": result.model,
        "stop_reason": result.stop_reason,
        "steps": result.steps,
        "tool_calls": len(calls),
        "tools_called": ";".join(result.tools_called),
        "expected_tools_used": expected.issubset(set(result.tools_called)),
        "invalid_tool_arguments": invalid_args,
        "final_answer_valid": bool(result.final_valid),
        "category_correct": fa.get("category") == request.get("expected_category"),
        "urgency_correct": fa.get("urgency") == request.get("expected_urgency"),
        "evidence_grounded": grounded,
        "latency_s": round(result.latency_s, 1),
    }


# ---------------------------------------------------------------- code-along progress
def code_along_status() -> dict:
    """Which CODE-ALONG gaps (and the extension) work right now, tested by behaviour.

    Used by 06_evidence_check.py. Writes nothing to the real action log.
    """
    import contextlib
    import io

    from src import config, safeguards, tools
    saved_log = config.ACTION_LOG
    config.ACTION_LOG = config.OUTPUT_DIR / "status_check_action_log.csv"
    tests = {
        "CA1": lambda: tools.read_status("Quantum Portal").get("error") == "unknown_service",
        "CA2": lambda: tools.read_status("printing").get("state") == "unknown",
        "CA3": lambda: tools.read_status("email").get("sensor_max") == 2080.0,
        "CA4": lambda: tools.search_knowledge("reset vpn token")["results"][0]["note_id"] == "KB-001",
        "CA5": lambda: tools.record_proposed_action("T", "a", "r").get("error") == "approval_required",
        "CA6": lambda: safeguards.within_step_limit(6, 6) is False and safeguards.within_step_limit(2, 6) is True,
        "CA7": lambda: (safeguards.request_approval("x", {}, "interactive", answer="maybe")[0] is False
                        and safeguards.request_approval("x", {}, "interactive", answer="y")[0] is True),
        "EXT": lambda: "sensor_trend" in tools.TOOL_FUNCTIONS
                       and tools.TOOL_FUNCTIONS["sensor_trend"]("hvac").get("direction") == "rising",
    }
    out = {}
    for name, test in tests.items():
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                out[name] = bool(test())
        except Exception:
            out[name] = False
    config.ACTION_LOG = saved_log
    out["switches_on"] = bool(config.STEP_LIMIT_ON and config.RESULT_VALIDATION_ON and config.APPROVAL_GATE_ON)
    return out
