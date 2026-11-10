"""The agent loop: observe, reason, act, learn, stop.

    1. OBSERVE  read the request
    2. REASON   ask the model what to do next
    3. ACT      if it asks for a tool: validate the arguments, check that a service it
                asks about is one the user mentioned, ask a human if the tool
                changes something, run the tool, validate its result
    4. OBSERVE  give the result back to the model and update the agent's state
    5. LEARN    when the model answers, check the answer against its schema and check
                that its evidence came from tool results it actually received; stop
                when the goal is met, the step limit is reached or something fails

No changes are needed in this file for Lab 1; the gaps to complete are in
src/tools.py and src/safeguards.py. Read it to understand what the trace shows.
"""
import json
import re
import time
from dataclasses import dataclass, field

from src import config
from src.data_store import get_request, load_status
from src.providers import ProviderError, make_session, model_name
from src.safeguards import request_approval, validate_tool_result, within_step_limit
from src.state import AgentState
from src.tool_schemas import validate_arguments, validate_final_answer
from src.tools import EXTRA_ALIASES, TOOL_FUNCTIONS, _find_service_row
from src.trace import TraceWriter

SYSTEM_PROMPT = f"""You are a service-desk triage agent for an enterprise's IT and facilities teams.
All systems are simulated; you work only through the tools you are given.

Always check with your tools before you answer; never answer from memory.

How to work:
1. Use read_status when the request names or describes a service, passing the name as the user wrote it;
   never swap in a different service. Use search_knowledge for how-to,
   policy and escalation questions. You may use both.
2. Call record_proposed_action only when the next step is an operational action: a restart, an
   escalation to a team, or a new incident. Do not call it for questions answered from the knowledge base.
   Record at most one action per request. A human must approve it. If the result is approval_denied,
   do not call it again.
3. Never invent a service status, a note or a fact. If a tool returns an error, or a status of "unknown",
   say so plainly and set needs_human to true.
4. When you have enough information, reply with ONLY one JSON object and no other text:
{{"request_id": "<id>", "category": "<one of {config.CATEGORIES}>", "urgency": "<one of {config.URGENCIES}>",
 "summary": "<one or two sentences>", "evidence": ["status:<service id>" or knowledge note ids such as "KB-001"],
 "next_step": "<what should happen next>", "action_recorded": <true|false>, "needs_human": <true|false>}}

Category guide: access (accounts, passwords, tokens, permissions); network (VPN connectivity, Wi-Fi);
application (business systems such as email, CRM, payroll, ERP, HR portal, video conferencing);
hardware (printers, laptops, peripherals); facilities (building systems, HVAC, server room);
information (policy and general questions).
Urgency guide: critical = safety risk or a business-critical service down for many teams; high = a service
down for a whole team or function, or an environmental alarm above threshold; medium = one user blocked or a
degraded service with a workaround; low = how-to questions and information requests.
Evidence may only cite tool results you actually received in this conversation."""

EXTENSION_PROMPT = ("\nYou also have sensor_trend: use it together with read_status when the direction of a "
                    "sensor reading (rising, falling, stable) matters for urgency, such as an environmental alarm.")


def system_prompt() -> str:
    return SYSTEM_PROMPT + (EXTENSION_PROMPT if config.EXTENSION_TOOL_ON else "")


PLAIN_PROMPT = "You are a helpful IT service-desk assistant. Answer the user's request briefly."




@dataclass
class AgentResult:
    request_id: str
    provider: str
    model: str
    final_answer: dict
    final_valid: bool
    stop_reason: str
    steps: int
    tools_called: list
    evidence_ids: list
    trace_path: str
    latency_s: float
    state: AgentState = field(repr=False, default=None)
    error: str = None             # the model error message, when stop_reason is model_error

    def show(self):
        print(f"\nRESULT  request={self.request_id}  provider={self.provider}  model={self.model}")
        print(f"  stop reason : {self.stop_reason}   steps: {self.steps}   time: {self.latency_s:.1f}s")
        print(f"  tools called: {self.tools_called}")
        print(f"  final answer valid: {self.final_valid}")
        if self.final_answer:
            print("  " + json.dumps(self.final_answer, indent=2).replace("\n", "\n  "))
        if self.error:
            print(f"  error: {self.error}")
        print(f"  trace: {self.trace_path}")


def format_request(request: dict) -> str:
    return (f"Request id: {request['id']}\n"
            f"Submitted: {request.get('submitted_at', 'unknown')} via {request.get('channel', 'portal')} "
            f"by the {request.get('requester_team', 'unknown')} team\n"
            f"Request: {request['text']}")


def _as_request(request) -> dict:
    if isinstance(request, dict):
        return request
    if re.fullmatch(r"[A-Za-z]\d{2,3}", str(request)):
        return get_request(str(request))
    return {"id": "ADHOC", "text": str(request)}


def parse_final_answer(text: str, request_id: str):
    """Extract a JSON object from the model's reply and check it. Returns (answer, problems)."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None, ["reply contains no JSON object"]
    try:
        answer = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return None, [f"reply is not valid JSON: {exc.msg}"]
    if isinstance(answer, dict):
        answer.setdefault("request_id", request_id)
    return answer, validate_final_answer(answer)


def _service_evidence_id(item):
    """Write status evidence with the service id the tool uses: "status:Campus Wi-Fi" and
    "status:payroll server" become "status:wifi" and "status:payroll". Other items are unchanged."""
    text = str(item)
    if not text.lower().startswith("status:"):
        return item
    row = _find_service_row(load_status(), text.split(":", 1)[1])
    return f"status:{row['service']}" if row is not None else item


def check_grounding(answer: dict, state: AgentState) -> list:
    """Did the answer come from the tools? Returns a list of problems ([] = grounded).

    A model can write a perfectly shaped answer without checking anything, and even
    cite "status:email" as evidence. This check compares the answer with the state:
    at least one tool must have been called, every evidence item must be one that
    a tool result actually returned in this run, and a status of "unknown" must be
    handed to a human (needs_human true) instead of being guessed. Status evidence may
    name the service in any way the tool understands; it is rewritten to the service id.
    """
    if not state.tool_history:
        return ["you answered without calling any tool"]
    received = {str(e).lower() for e in state.evidence_ids}
    cited = answer.get("evidence") or []
    if isinstance(cited, list):
        cited = answer["evidence"] = [_service_evidence_id(e) for e in cited]
    made_up = [e for e in cited if str(e).lower() not in received]
    if made_up:
        return [f"evidence {made_up} did not come from any tool result you received "
                f"(evidence you may cite: {state.evidence_ids or 'none'})"]
    unknown = [t["result"]["service"] for t in state.tool_history
               if t["name"] == "read_status" and isinstance(t["result"], dict) and t["result"].get("state") == "unknown"]
    if unknown and answer.get("needs_human") is not True:
        return [f"read_status returned state 'unknown' for {unknown}; say that the status is unknown "
                f"and set needs_human to true"]
    return []


def _say(verbose, text):
    if verbose:
        print(text)


def _parse_arguments(call):
    if call.arguments is not None:
        return call.arguments, None
    try:
        args = json.loads(call.arguments_raw or "{}")
        return args, None
    except (json.JSONDecodeError, TypeError) as exc:
        return None, f"arguments are not valid JSON: {exc}"


def check_service_named(service: str, request_text: str):
    """Is the service the model asked about one that the request mentions?

    A model may replace a name it does not recognise ("Quantum Portal") with a real
    service that sounds similar ("hr_portal"). The tool would then return a correct
    status for the wrong service, and no check on the result could notice. So before
    the tool runs, the agent checks that one of the service's names or aliases
    appears in the request. Returns an error dict for the model, or None if all is well.
    Names the tool does not know are let through: the tool reports unknown_service.
    """
    row = _find_service_row(load_status(), service)
    if row is None:
        return None
    names = {row["service"], row["service"].replace("_", " "), str(row["display_name"]).lower()}
    names |= {a.strip() for a in str(row["aliases"]).split(";") if a.strip()}
    names |= {alias for alias, target in EXTRA_ALIASES.items() if target == row["service"]}
    text = request_text.lower()
    if any(re.search(rf"(?<![a-z0-9]){re.escape(n)}(?![a-z0-9])", text) for n in names):
        return None
    return {"error": "service_not_in_request", "service": service,
            "message": f"The request does not mention '{service}'. Pass the service name exactly as the "
                       "user wrote it, even if you do not recognise it."}


def execute_tool_call(call, state: AgentState, trace: TraceWriter, step: int,
                      approval_mode: str, verbose: bool) -> dict:
    """ACT then OBSERVE for one tool call. Always returns a dict for the model."""
    args, parse_error = _parse_arguments(call)
    trace.log("tool_call", stage="act", step=step, name=call.name, arguments=args if args is not None else call.arguments_raw)
    _say(verbose, f"  [act]     {call.name}({json.dumps(args) if args is not None else call.arguments_raw})")

    if call.name not in TOOL_FUNCTIONS:
        result = {"error": "unknown_tool", "message": f"There is no tool called '{call.name}'.",
                  "valid_tools": list(TOOL_FUNCTIONS)}
    elif parse_error:
        trace.log("malformed_tool_call", stage="act", step=step, error=parse_error)
        result = {"error": "malformed_arguments", "message": parse_error}
    else:
        problems = validate_arguments(call.name, args)
        if problems:
            trace.log("invalid_arguments", stage="act", step=step, name=call.name, problems=problems)
            result = {"error": "invalid_arguments", "problems": problems,
                      "message": "Fix the arguments to match the tool's contract and try again."}
        else:
            wrong_service = (check_service_named(args["service"], state.request_text)
                             if call.name in ("read_status", "sensor_trend") else None)
            if wrong_service:
                trace.log("service_not_in_request", stage="act", step=step, name=call.name,
                          error=wrong_service["message"])
                result = wrong_service
            else:
                result = _run_tool(call.name, args, state, trace, step, approval_mode)

    # Safeguard 2: tool-result validation (only when switched on in config.py).
    problems = validate_tool_result(call.name, result) if config.RESULT_VALIDATION_ON else []
    if problems:
        trace.log("tool_result_invalid", stage="observe", step=step, name=call.name, problems=problems, raw=result)
        result = {"error": "invalid_tool_result", "problems": problems}

    state.record_tool(call.name, args, result)
    trace.log("tool_result", stage="observe", step=step, name=call.name, result=result)
    _say(verbose, f"  [observe] {json.dumps(result, default=str)[:220]}")
    return result


def _run_tool(name, args, state, trace, step, approval_mode):
    try:
        if name == "record_proposed_action":
            recorded = [e for e in state.evidence_ids if str(e).startswith("ACT-")]
            if recorded:   # one action per request: never ask a human, or log, twice
                trace.log("duplicate_action", stage="act", step=step, error=f"already recorded: {recorded[0]}")
                return {"error": "action_already_recorded",
                        "message": f"An action ({recorded[0]}) is already recorded for this request. "
                                   "Do not record another; give your final answer."}
            # Safeguard 3: human approval (only when switched on in config.py).
            if config.APPROVAL_GATE_ON:
                approved, approver = request_approval(name, args, approval_mode)
            else:
                approved, approver = True, "nobody (approval gate switched off)"
            state.approvals.append({"action": args.get("action"), "approved": approved, "approver": approver})
            trace.log("approval", stage="act", step=step, approved=approved, approver=approver, action=args.get("action"))
            if not approved:
                return {"error": "approval_denied",
                        "message": "A human reviewer declined this action. Do not try it again; "
                                   "report that the request needs human follow-up."}
            return TOOL_FUNCTIONS[name](**args, approved_by=approver)
        return TOOL_FUNCTIONS[name](**args)
    except NotImplementedError as exc:
        trace.log("tool_not_implemented", stage="act", step=step, error=str(exc))
        print(f"\n  !! {exc}\n")
        return {"error": "tool_not_implemented", "message": str(exc)}
    except Exception as exc:
        trace.log("tool_exception", stage="act", step=step, error=f"{type(exc).__name__}: {exc}")
        return {"error": "tool_exception", "message": f"{type(exc).__name__}: {exc}"}


def run_agent(request, provider: str = "open", approval_mode: str = "interactive",
              max_steps: int = config.MAX_STEPS, verbose: bool = True) -> AgentResult:
    """Run the agent on one request (an id such as "R01", a request dict, or plain text)."""
    request = _as_request(request)
    rid = request["id"]
    state = AgentState(request_id=rid, request_text=request["text"])
    trace = TraceWriter(rid, provider)
    started = time.time()

    trace.log("run_start", stage="observe", step=0, request_id=rid, request_text=request["text"],
              provider=provider, model=model_name(provider), max_steps=max_steps, approval_mode=approval_mode,
              safeguards={"step_limit": config.STEP_LIMIT_ON, "result_validation": config.RESULT_VALIDATION_ON,
                          "approval_gate": config.APPROVAL_GATE_ON}, extension_tool=config.EXTENSION_TOOL_ON)
    _say(verbose, f"\n=== {rid} via {provider} ({model_name(provider)}) ===\n  [observe] {request['text']}")

    try:
        session = make_session(provider, system_prompt())
    except Exception as exc:
        trace.log("model_error", stage="reason", step=0, error=str(exc))
        state.stop_reason = "model_error"
        return _finish(state, trace, provider, started, final_valid=False, verbose=verbose, error=str(exc))
    session.add_user(format_request(request))

    final_retries = 0
    grounding_retries = 0
    final_valid = False
    model_error = None
    while True:
        # Safeguard 1 (yours) and the course safety net (always on).
        if config.STEP_LIMIT_ON:
            try:
                allowed = within_step_limit(state.steps, max_steps)
            except NotImplementedError as exc:
                print(f"\n  !! {exc}\n")
                trace.log("safeguard_not_implemented", stage="reason", step=state.steps, error=str(exc))
                state.stop_reason = "step_limit_not_written"
                break
            if not allowed:
                state.stop_reason = "step_limit"
                break
        if state.steps >= config.COURSE_SAFETY_NET_STEPS:
            state.stop_reason = "course_safety_net"
            break
        state.steps += 1
        step = state.steps

        try:
            turn = session.next_turn()                                     # REASON
        except ProviderError as exc:
            trace.log("model_error", stage="reason", step=step, error=str(exc))
            _say(verbose, f"  [reason]  model error: {exc}")
            state.stop_reason, model_error = "model_error", str(exc)
            break

        trace.log("model_turn", stage="reason", step=step, latency_s=round(turn.latency_s, 2), text=turn.text,
                  tool_calls=[{"name": c.name, "arguments": c.arguments if c.arguments is not None else c.arguments_raw}
                              for c in turn.tool_calls], note=turn.note)
        if turn.note:
            _say(verbose, f"  [reason]  note: {turn.note}")

        if turn.tool_calls:
            _say(verbose, f"  [reason]  step {step}: model asks for {', '.join(c.name for c in turn.tool_calls)}")
            results = [(c, execute_tool_call(c, state, trace, step, approval_mode, verbose)) for c in turn.tool_calls]
            session.add_tool_results(results)
            continue

        answer, problems = parse_final_answer(turn.text, rid)               # LEARN
        if not problems:
            ungrounded = check_grounding(answer, state)
            if ungrounded:
                trace.log("answer_rejected", stage="learn", step=step, problems=ungrounded, answer=answer)
                _say(verbose, f"  [learn]   step {step}: answer rejected: {ungrounded}")
                state.final_answer = answer
                if grounding_retries < config.MAX_FINAL_ANSWER_RETRIES:
                    grounding_retries += 1
                    fix = ("Call the tools you need first, then reply with ONLY the JSON object."
                           if not state.tool_history else
                           "Correct the answer and reply with ONLY the JSON object. Do not repeat tool calls "
                           "or actions you have already made.")
                    session.add_user("Your answer was rejected: " + "; ".join(ungrounded) + ". " + fix)
                    continue
                state.stop_reason = "answer_rejected"
                break
            state.final_answer, final_valid = answer, True
            trace.log("final_answer", stage="learn", step=step, answer=answer, state=state.summary())
            _say(verbose, f"  [learn]   step {step}: valid final answer")
            state.stop_reason = "goal_met"
            break
        trace.log("invalid_final_answer", stage="learn", step=step, problems=problems, text=turn.text)
        _say(verbose, f"  [learn]   step {step}: final answer invalid: {problems}")
        state.final_answer = answer
        if final_retries < config.MAX_FINAL_ANSWER_RETRIES:
            final_retries += 1
            session.add_user("Your last reply was not a valid final answer: " + "; ".join(problems) +
                             ". Reply again with ONLY the JSON object described in your instructions.")
            continue
        state.stop_reason = "invalid_final_answer"
        break

    return _finish(state, trace, provider, started, final_valid, verbose, error=model_error)


def _finish(state, trace, provider, started, final_valid, verbose, error=None):
    trace.log("stop", stage="learn", step=state.steps, reason=state.stop_reason, steps=state.steps, error=error)
    result = AgentResult(request_id=state.request_id, provider=provider, model=model_name(provider),
                         final_answer=state.final_answer, final_valid=final_valid, stop_reason=state.stop_reason,
                         steps=state.steps, tools_called=state.tools_called, evidence_ids=state.evidence_ids,
                         trace_path=str(trace.path), latency_s=time.time() - started, state=state, error=error)
    if verbose:
        result.show()
    return result


def run_plain(request, provider: str = "open") -> str:
    """The same request answered by the model alone: no tools, no state, no checks."""
    request = _as_request(request)
    session = make_session(provider, PLAIN_PROMPT, use_tools=False)
    session.add_user(request["text"])
    return session.next_turn().text
