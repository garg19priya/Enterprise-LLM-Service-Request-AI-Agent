"""Test what each safeguard stops.

    %run scripts/try_safeguards.py

Runs three small experiments. They use the rule-based mock model, so they are
quick and free; use 03_run_agent.py afterwards to see the same with the open model.
Run it once BEFORE you switch the safeguards on and once AFTER, and compare.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Reload the project code on every run, so edits to src/*.py are picked up.
for _m in [m for m in list(sys.modules) if m == "src" or m.startswith("src.")]:
    del sys.modules[_m]

from src import config  # noqa: E402
from src import safeguards  # noqa: E402
from src.agent import run_agent  # noqa: E402
from src.tools import count_logged_actions  # noqa: E402


def switch(name):
    return "ON " if getattr(config, name) else "OFF"


def experiment_step_limit():
    print(f"\n1) STEP LIMIT          switch STEP_LIMIT_ON = {switch('STEP_LIMIT_ON')}")
    try:
        cases = [(0, 6, True), (5, 6, True), (6, 6, False), (9, 6, False)]
        wrong = [(s, m) for s, m, expected in cases if safeguards.within_step_limit(s, m) is not expected]
        print("   within_step_limit:", "correct for steps 0, 5, 6 and 9 of 6" if not wrong
              else f"WRONG for (steps, max) = {wrong}")
    except NotImplementedError as exc:
        print(f"   NOT YET -> {exc}")
    result = run_agent("R01", provider="mock", approval_mode="approve", max_steps=1, verbose=False)
    if result.stop_reason == "step_limit":
        print("   agent with max_steps=1: STOPPED by the step limit after 1 step (no final answer)")
    elif result.stop_reason == "step_limit_not_written":
        print("   agent with max_steps=1: the switch is on but CODE-ALONG 6 is not written, so the agent stopped at once")
    else:
        print(f"   agent with max_steps=1: NOT stopped. It ran {result.steps} steps (stop reason: {result.stop_reason}),"
              " because the step limit is off.")


def experiment_validation():
    print(f"\n2) RESULT VALIDATION   switch RESULT_VALIDATION_ON = {switch('RESULT_VALIDATION_ON')}")
    broken = {"service": "email", "state": "probably fine"}
    problems = safeguards.validate_tool_result("read_status", broken)
    print(f"   a broken read_status result {broken}")
    print("   validate_tool_result finds:")
    for p in problems:
        print(f"     - {p}")
    if config.RESULT_VALIDATION_ON:
        print("   -> with the switch ON, the agent replaces it with an invalid_tool_result error,")
        print("      so the model never sees the made-up state.")
    else:
        print("   -> with the switch OFF, the agent would pass this straight to the model.")
    print("   an error such as {'error': 'unknown_service'} is VALID:",
          safeguards.validate_tool_result("read_status", {"error": "unknown_service"}) == [])


def experiment_approval():
    print(f"\n3) APPROVAL GATE       switch APPROVAL_GATE_ON = {switch('APPROVAL_GATE_ON')}")
    expected = {"y": True, "yes": True, " YES ": True, "n": False, "": False, "maybe": False, "yes please": False}
    try:
        import io
        import contextlib
        wrong = []
        for answer, should in expected.items():
            with contextlib.redirect_stdout(io.StringIO()):
                approved, _ = safeguards.request_approval("record_proposed_action", {}, "interactive", answer=answer)
            if approved is not should:
                wrong.append(answer)
        print("   answers y, yes, ' YES ' approve; n, '', maybe, 'yes please' are DENIED:",
              "correct" if not wrong else f"WRONG for {wrong}")
    except NotImplementedError as exc:
        print(f"   NOT YET -> {exc}")

    config.ACTION_LOG = config.OUTPUT_DIR / "try_safeguards_action_log.csv"
    before = count_logged_actions()
    result = run_agent("R02", provider="mock", approval_mode="deny", verbose=False)
    written = count_logged_actions() - before
    who = result.state.approvals[0]["approver"] if result.state.approvals else "nobody asked"
    print(f"   R02 'Restart the payroll server now', reviewer would say NO: approver = {who}")
    print(f"   actions written to the log: {written}  ->  " +
          ("BLOCKED, as it should be" if written == 0 else "WRITTEN WITHOUT APPROVAL. Switch the gate on."))


def main():
    print("What each safeguard stops (mock model, no API calls)")
    experiment_step_limit()
    experiment_validation()
    experiment_approval()
    print("\nNext: %run scripts/03_run_agent.py --provider open --request R02 --approval interactive")


if __name__ == "__main__":
    main()
