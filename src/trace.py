"""Execution trace: one JSON line per event, labelled with its stage in the
observe, reason, act, learn cycle.

    observe  the agent takes in the request or a tool result
    reason   the model decides what to do next
    act      the agent calls a tool (after validation and, if needed, approval)
    learn    the agent updates its state and checks its final answer
"""
import json
from datetime import datetime
from pathlib import Path

from src import config


class TraceWriter:
    def __init__(self, request_id: str, provider: str, directory: Path = None):
        directory = Path(directory or config.TRACE_DIR)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        self.path = directory / f"{request_id}_{provider}_{stamp}.jsonl"
        self.events = []

    def log(self, event: str, stage: str = None, step: int = None, **data) -> dict:
        record = {"time": datetime.now().isoformat(timespec="milliseconds"), "step": step,
                  "stage": stage, "event": event, **data}
        self.events.append(record)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
        return record


def load_trace(path) -> list:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _short(value, width):
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 3] + "..."


def describe(event: dict, width: int = 90) -> str:
    e = event["event"]
    if e == "run_start":
        return _short(f"request: {event.get('request_text')}", width)
    if e == "model_turn":
        if event.get("tool_calls"):
            calls = ", ".join(f"{c['name']}({_short(c.get('arguments'), 50)})" for c in event["tool_calls"])
            return _short(f"model asks for {calls}", width)
        return _short(f"model replies: {event.get('text', '')}", width)
    if e == "tool_call":
        return _short(f"{event.get('name')} {json.dumps(event.get('arguments'), default=str)}", width)
    if e == "tool_result":
        return _short(f"{event.get('name')} -> {json.dumps(event.get('result'), default=str)}", width)
    if e in ("invalid_arguments", "tool_result_invalid", "invalid_final_answer", "answer_rejected",
             "service_not_in_request", "duplicate_action", "model_error", "malformed_tool_call",
             "tool_not_implemented"):
        return _short(f"{event.get('problems') or event.get('error')}", width)
    if e == "approval":
        verdict = "APPROVED" if event.get("approved") else "DENIED"
        return _short(f"{verdict} by {event.get('approver')}: {event.get('action')}", width)
    if e == "final_answer":
        return _short(event.get("answer"), width)
    if e == "stop":
        return f"reason: {event.get('reason')} after {event.get('steps')} step(s)"
    return _short({k: v for k, v in event.items() if k not in ("time", "step", "stage", "event")}, width)


def show_trace(trace, width: int = 90) -> None:
    """Print a readable view of a trace (a path, or a list of events)."""
    events = load_trace(trace) if isinstance(trace, (str, Path)) else trace
    print(f"{'step':>4}  {'stage':<8} {'event':<22} detail")
    print("-" * (40 + width))
    for ev in events:
        step = "" if ev.get("step") is None else ev["step"]
        print(f"{step!s:>4}  {(ev.get('stage') or ''):<8} {ev['event']:<22} {describe(ev, width)}")
