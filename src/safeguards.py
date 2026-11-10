"""Safeguards that make the agent dependable.

  1. Switch each safeguard on in src/config.py (STEP_LIMIT_ON, RESULT_VALIDATION_ON,
     APPROVAL_GATE_ON).
  2. Complete the two short gaps below: CODE-ALONG 6 (step limit) and
     CODE-ALONG 7 (reading the human's answer). Safeguard 2 is already written:
     read it and see what it checks.
  3. Test what each one stops:  %run scripts/try_safeguards.py

While a switch is off, the agent behaves as if that safeguard did not exist: no step
limit (only the course safety net of 15 steps), tool results trusted blindly, and
every action approved automatically. Acceptance case T4 fails until the approval
gate is on and working.
"""
from src.config import MAX_STEPS

VALID_STATES = {"operational", "degraded", "outage", "maintenance", "unknown"}


# ----------------------------------------------------------------------------
# Safeguard 1: step limit                      (switch: STEP_LIMIT_ON)
# ----------------------------------------------------------------------------
def within_step_limit(steps_taken: int, max_steps: int = MAX_STEPS) -> bool:
    """Return True if the agent may take another step, False if it must stop.

    The agent calls this before every model turn. steps_taken starts at 0.
    """
    # ------------------------------------------------------------------
    # CODE-ALONG 6: the step limit.  (1 line)
    # WHY  A model can loop: call a tool, call it again, and never answer.
    #      A step limit guarantees the agent always stops.
    # DO   Return True while steps_taken is LESS THAN max_steps, else False.
    # HINT A comparison such as  a < b  is already True or False, so you can
    #      return it directly.  With max_steps = 6, steps 0 to 5 are allowed.
    # CHECK %run scripts/try_safeguards.py   (the step-limit test should say STOPPED)
    # ------------------------------------------------------------------
    raise NotImplementedError("CODE-ALONG 6 (src/safeguards.py): return whether steps_taken < max_steps")


# ----------------------------------------------------------------------------
# Safeguard 2: tool-result validation          (switch: RESULT_VALIDATION_ON)
# Already written: read it, then switch it on.
# ----------------------------------------------------------------------------
def validate_tool_result(tool_name: str, result) -> list:
    """Check a tool's output against its contract. Return a list of problems ([] = valid).

    An error dict such as {"error": "unknown_service"} is a VALID result: the tool
    reported a problem in the agreed way. Invalid means the tool broke its contract.
    """
    if not isinstance(result, dict):
        return [f"{tool_name} must return a dict, got {type(result).__name__}"]
    if "error" in result:
        return [] if isinstance(result["error"], str) else ["'error' must be a string"]

    problems = []
    if tool_name == "read_status":
        for key in ("service", "state", "sensor_mean", "sensor_max", "threshold", "above_threshold"):
            if key not in result:
                problems.append(f"read_status result is missing '{key}'")
        if result.get("state") not in VALID_STATES:
            problems.append(f"state '{result.get('state')}' is not one of {sorted(VALID_STATES)}")
    elif tool_name == "search_knowledge":
        hits = result.get("results")
        if not isinstance(hits, list):
            problems.append("search_knowledge result must contain a 'results' list")
        else:
            for hit in hits:
                if not str(hit.get("note_id", "")).startswith("KB-"):
                    problems.append(f"result without a valid note_id: {hit}")
    elif tool_name == "record_proposed_action":
        if not str(result.get("action_id", "")).startswith("ACT-"):
            problems.append("record_proposed_action result must contain an action_id")
    elif tool_name == "sensor_trend":
        if result.get("direction") not in ("rising", "falling", "stable"):
            problems.append("sensor_trend direction must be rising, falling or stable")
    return problems


# ----------------------------------------------------------------------------
# Safeguard 3: simulated human approval        (switch: APPROVAL_GATE_ON)
# ----------------------------------------------------------------------------
def request_approval(tool_name: str, arguments: dict, mode: str = "interactive", answer: str = None):
    """Ask a human before an action is recorded. Return (approved: bool, approver: str).

    mode
      "interactive"  ask in the notebook: type y to approve, anything else to deny
      "approve"      approve automatically (used for benchmarks)
      "deny"         deny automatically (used by acceptance case T4)
    answer           a ready-made answer (used by try_safeguards.py instead of typing)
    """
    if mode == "approve":
        return True, "auto-approve (test mode)"
    if mode == "deny":
        return False, "auto-deny (test mode)"

    print("\n" + "=" * 70)
    print("HUMAN APPROVAL NEEDED")
    print(f"  Tool:    {tool_name}")
    print(f"  Request: {arguments.get('request_id')}")
    print(f"  Action:  {arguments.get('action')}")
    print(f"  Reason:  {arguments.get('reason')}")
    if answer is None:
        answer = input("Approve this action? [y/N] ")
    answer = answer.strip().lower()                # e.g. " Yes " -> "yes"

    # ------------------------------------------------------------------
    # CODE-ALONG 7: approve only on a clear yes.  (1 line)
    # WHY  A safe gate treats anything that is not a clear yes as a no:
    #      an empty answer, a typo or "maybe" must never approve an action.
    # DO   Create a variable `approved`: True only if `answer` is "y" or "yes".
    # HINT `x in ("a", "b")` is True when x equals either value. `answer` is
    #      already lower case with spaces removed.
    # CHECK %run scripts/try_safeguards.py   ("maybe" and "" must be DENIED)
    # ------------------------------------------------------------------
    raise NotImplementedError("CODE-ALONG 7 (src/safeguards.py): set approved from the answer")

    print("  -> approved" if approved else "  -> denied")
    print("=" * 70)
    return approved, ("human reviewer" if approved else "human reviewer (denied)")
