"""Run the four acceptance cases (target: at least 3 of 4).

    %run scripts/05_run_tests.py --provider open
    %run scripts/05_run_tests.py --provider open --extra   # also the missing-status case R04

Writes outputs/test_report.md.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Reload the project code on every run, so edits to src/*.py are picked up.
for _m in [m for m in list(sys.modules) if m == "src" or m.startswith("src.")]:
    del sys.modules[_m]

from src import config  # noqa: E402
from src.evaluation import ACCEPTANCE_CASES, EXTRA_CASES, run_case  # noqa: E402
from src.providers import model_name  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="Lab 1 acceptance tests")
    parser.add_argument("--provider", default="open", choices=["open", "gemini", "gemini_openai", "mock"])
    parser.add_argument("--extra", action="store_true", help="also run the missing-status case")
    args = parser.parse_args(argv)

    cases = ACCEPTANCE_CASES + (EXTRA_CASES if args.extra else [])
    outcomes = []
    for case in cases:
        print(f"Running {case[0]} ({case[4]}) on {case[1]}...", flush=True)
        outcome = run_case(case, args.provider)
        outcomes.append(outcome)
        print(f"  {'PASS' if outcome['passed'] else 'FAIL'}" +
              ("" if outcome["passed"] else "  - " + "; ".join(outcome["reasons"])))

    core = [o for o in outcomes if o["case"].startswith("T")]
    passed = sum(o["passed"] for o in core)
    verdict = "TARGET MET" if passed >= 3 else "TARGET NOT MET"
    lines = ["# Lab 1 acceptance test report", "",
             f"Provider: {args.provider} ({model_name(args.provider)})  ",
             f"Run at: {datetime.now().isoformat(timespec='seconds')}  ",
             f"Result: **{passed} of 4 acceptance cases passed ({verdict}; target 3 of 4)**", "",
             "| Case | Request | What it tests | Result | Notes |", "| --- | --- | --- | --- | --- |"]
    for o in outcomes:
        notes = "; ".join(o["reasons"]) if o["reasons"] else f"tools: {', '.join(o['tools_called']) or 'none'}"
        lines.append(f"| {o['case']} | {o['request_id']}: {o['request_text']} | {o['description']} | "
                     f"{'PASS' if o['passed'] else 'FAIL'} | {notes.replace('|', '/')} |")
    lines += ["", "Traces:"] + [f"- {o['case']}: {Path(o['trace']).name}" for o in outcomes]
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUTPUT_DIR / "test_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{passed} of 4 passed: {verdict}. Saved outputs/test_report.md")


if __name__ == "__main__":
    main(sys.argv[1:])
