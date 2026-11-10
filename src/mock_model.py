"""A rule-based stand-in for a language model. It is NOT an LLM.

Use it with --provider mock when no GPU or API is available, so the tools,
trace, safeguards and tests can still be exercised. It is also a quick way to
check the project end to end without a GPU or an API key.

It reads the conversation in OpenAI chat format and returns either tool calls
or a final JSON answer, following simple keyword rules.
"""
import json
import re

from src.data_store import load_status

HOWTO_PATTERNS = ["how do i", "locked out", "paper jam", "password"]
ACTION_WORDS = ["restart"]
SERVICE_CATEGORY = {"email": "application", "crm": "application", "payroll": "application", "erp": "application",
                    "hr_portal": "application", "video": "application", "file_share": "application",
                    "vpn": "network", "wifi": "network", "sso": "access", "printing": "hardware", "hvac": "facilities"}


def _request_parts(messages):
    first_user = next(m["content"] for m in messages if m["role"] == "user")
    rid = re.search(r"Request id:\s*(\S+)", first_user)
    text = re.search(r"Request:\s*(.+)", first_user, flags=re.DOTALL)
    return (rid.group(1) if rid else "UNKNOWN"), (text.group(1).strip() if text else first_user)


def _tool_results(messages):
    """[(tool_name, arguments, result_dict)] in order."""
    names = {}
    for m in messages:
        for call in m.get("tool_calls") or []:
            names[call["id"]] = (call["function"]["name"], json.loads(call["function"]["arguments"] or "{}"))
    out = []
    for m in messages:
        if m["role"] == "tool":
            name, args = names.get(m.get("tool_call_id"), ("unknown", {}))
            try:
                result = json.loads(m["content"])
            except (TypeError, json.JSONDecodeError):
                result = {"raw": m["content"]}
            out.append((name, args, result))
    return out


def _detect_service(text):
    lowered = text.lower()
    best, best_len = None, 0
    for _, row in load_status().iterrows():
        for alias in [row["service"]] + str(row["aliases"]).split(";"):
            alias = alias.strip()
            if alias and re.search(rf"\b{re.escape(alias)}\b", lowered) and len(alias) > best_len:
                best, best_len = row["service"], len(alias)
    if best:
        return best
    match = re.search(r"is (?:the )?(.+?) (?:working|down|up|available)", lowered)
    return match.group(1).strip() if match else None


def _keywords(text):
    words = re.findall(r"[a-z0-9-]+", text.lower())
    drop = {"how", "do", "i", "my", "the", "a", "to", "is", "are", "we", "in", "of", "can", "get", "this", "on", "has"}
    return " ".join(w for w in words if w not in drop)[:80]


def _call(name, args, n):
    return {"id": f"mock-call-{n}", "name": name, "arguments": args}


def mock_reply(messages, tools_enabled=True) -> dict:
    """Return {"content": str|None, "tool_calls": [{"id","name","arguments"}]}."""
    request_id, text = _request_parts(messages)
    lowered = text.lower()

    if not tools_enabled:
        return {"content": (f"It sounds like a problem with {(_detect_service(text) or 'that system')}. "
                            "It is probably working normally; try restarting your device and signing in again. "
                            "If that does not help, wait a while and try later."),
                "tool_calls": []}

    # A correction request after an invalid final answer: just answer again.
    results = _tool_results(messages)
    # A call rejected for bad arguments does not count: retry it, as a real model would.
    called = [r[0] for r in results if r[2].get("error") not in ("invalid_arguments", "malformed_arguments")]
    n = len(called) + 1
    service = _detect_service(text)
    is_howto = any(p in lowered for p in HOWTO_PATTERNS)

    status = next((r[2] for r in reversed(results) if r[0] == "read_status"
                   and r[2].get("error") not in ("invalid_arguments", "malformed_arguments")), None)
    if not is_howto and service and "read_status" not in called:
        return {"content": None, "tool_calls": [_call("read_status", {"service": service}, n)]}
    if is_howto and "search_knowledge" not in called:
        return {"content": None, "tool_calls": [_call("search_knowledge", {"query": _keywords(text), "top_k": 3}, n)]}

    status_ok = isinstance(status, dict) and "error" not in status and status.get("state") != "unknown"
    wide_impact = any(w in lowered for w in ["whole", "everyone"])
    wants_action = any(w in lowered for w in ACTION_WORDS) or (
        status_ok and (status.get("state") == "outage" and wide_impact or
                       (status.get("service") == "hvac" and status.get("above_threshold") and "alarm" in lowered)))
    if wants_action and status_ok and "record_proposed_action" not in called:
        action = (f"Restart {status['display_name']} under an approved change" if "restart" in lowered
                  else f"Escalate to the {status['owner_team']} team as a high-priority incident")
        reason = f"status:{status['service']} is {status['state']}; requester reported: {text[:80]}"
        return {"content": None, "tool_calls": [_call("record_proposed_action",
                                                      {"request_id": request_id, "action": action, "reason": reason}, n)]}

    # ---- final answer
    evidence, needs_human, action_recorded = [], False, False
    for name, _args, result in results:
        if result.get("error") in ("invalid_arguments", "malformed_arguments"):
            continue
        if "error" in result:
            needs_human = True
            continue
        if name == "read_status":
            evidence.append(f"status:{result['service']}")
            if result.get("state") == "unknown":
                needs_human = True
        elif name == "search_knowledge":
            evidence += [h["note_id"] for h in result.get("results", [])[:1]]
        elif name == "record_proposed_action":
            action_recorded = True

    svc_key = status.get("service") if status_ok else None
    category = ("access" if any(w in lowered for w in ["token", "password", "locked"]) else
                SERVICE_CATEGORY.get(svc_key or service, "application"))
    if any(w in lowered for w in ["whole", "restart", "alarm"]) or (status_ok and status.get("state") == "outage"):
        urgency = "high"
    elif is_howto and "locked" not in lowered:
        urgency = "low"
    else:
        urgency = "medium"

    if status is not None and "error" in status:
        summary = f"'{service}' is not a service I can find, so I cannot report its status."
        next_step = "Ask the requester which system they mean and pass the request to a human agent."
    elif status is not None and status.get("state") == "unknown":
        summary = f"The status of {status['display_name']} is not recorded, so it is unknown."
        next_step = f"Ask the {status['owner_team']} team to confirm the service status."
    elif status_ok:
        summary = f"{status['display_name']} is {status['state']} (sensor max {status['sensor_max']}, threshold {status['threshold']})."
        next_step = f"Follow up with the {status['owner_team']} team."
    else:
        hits = next((r[2].get("results", []) for r in results if r[0] == "search_knowledge"), [])
        summary = f"See {hits[0]['note_id']}: {hits[0]['title']}." if hits else "No matching guidance was found."
        next_step = "Send the requester the steps from the knowledge note." if hits else "Pass to a human agent."
        needs_human = needs_human or not hits
    if any(r[0] == "record_proposed_action" and r[2].get("error") == "approval_denied" for r in results):
        next_step = "The action was not approved; a human must decide the next step."
        needs_human = True

    answer = {"request_id": request_id, "category": category, "urgency": urgency, "summary": summary,
              "evidence": evidence, "next_step": next_step, "action_recorded": action_recorded,
              "needs_human": needs_human}
    return {"content": json.dumps(answer), "tool_calls": []}
