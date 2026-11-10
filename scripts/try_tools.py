"""Check your tools directly, without any model. Run it after every CODE-ALONG gap.

    %run scripts/try_tools.py                      # all checks, in code-along order
    %run scripts/try_tools.py --service "teams call"   # try one name (guided alias change)
    %run scripts/try_tools.py --extension          # check the optional sensor_trend tool

Each check shows what goes in, what comes out, and OK or what is still missing.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Reload the project code on every run, so edits to src/*.py are picked up.
for _m in [m for m in list(sys.modules) if m == "src" or m.startswith("src.")]:
    del sys.modules[_m]

from src import config  # noqa: E402
from src.safeguards import validate_tool_result  # noqa: E402
from src.tools import TOOL_FUNCTIONS  # noqa: E402

# (gap, tool, arguments, what you should see, check on the result)
CHECKS = [
    ("CODE-ALONG 1", "read_status", {"service": "Quantum Portal"}, "error unknown_service, nothing invented",
     lambda r: r.get("error") == "unknown_service" and "known_services" in r),
    ("CODE-ALONG 2", "read_status", {"service": "printing"}, "state 'unknown' (the cell is empty on purpose)",
     lambda r: r.get("state") == "unknown"),
    ("CODE-ALONG 3", "read_status", {"service": "email"}, "state outage, sensor_max 2080.0, above_threshold True",
     lambda r: r.get("state") == "outage" and r.get("sensor_max") == 2080.0 and r.get("above_threshold") is True),
    ("CODE-ALONG 3", "read_status", {"service": "Wi-Fi"}, "an alias resolves to wifi, state degraded",
     lambda r: r.get("service") == "wifi" and r.get("state") == "degraded"),
    ("CODE-ALONG 4", "search_knowledge", {"query": "reset vpn token"}, "KB-001 ranked first",
     lambda r: bool(r.get("results")) and r["results"][0]["note_id"] == "KB-001"),
    ("CODE-ALONG 4", "search_knowledge", {"query": "zebra migration", "top_k": 2}, "an empty results list",
     lambda r: r.get("results") == []),
    ("CODE-ALONG 5", "record_proposed_action", {"request_id": "TEST", "action": "Test action", "reason": "Trying the tool"},
     "error approval_required (no approver given)", lambda r: r.get("error") == "approval_required"),
]

EXTENSION_CHECKS = [
    ("EXTENSION", "sensor_trend", {"service": "hvac"}, "direction rising", lambda r: r.get("direction") == "rising"),
    ("EXTENSION", "sensor_trend", {"service": "vpn"}, "direction stable", lambda r: r.get("direction") == "stable"),
    ("EXTENSION", "sensor_trend", {"service": "erp", "hours": 6}, "direction falling",
     lambda r: r.get("direction") == "falling"),
    ("EXTENSION", "sensor_trend", {"service": "Quantum Portal"}, "error unknown_service",
     lambda r: r.get("error") == "unknown_service"),
]


def run_checks(checks):
    done = 0
    for gap, name, args, expected, ok in checks:
        print(f"\n[{gap}] {name}({json.dumps(args)})\n  you should see: {expected}")
        try:
            result = TOOL_FUNCTIONS[name](**args)
        except NotImplementedError as exc:
            print(f"  NOT YET -> {exc}")
            continue
        except Exception as exc:
            print(f"  CRASHED -> {type(exc).__name__}: {exc}\n  Read the error, check your line, save, and rerun.")
            continue
        print("  got:  " + json.dumps(result, default=str)[:300])
        if name == "read_status" and "error" not in result and result.get("sensor_max") is None:
            print("  NOT YET -> CODE-ALONG 3 (src/tools.py): mean and peak are still the placeholder None")
            if gap == "CODE-ALONG 3":
                continue
        problems = validate_tool_result(name, result)
        if problems:
            print("  CONTRACT BROKEN -> " + "; ".join(problems))
        elif ok(result):
            print("  OK")
            done += 1
        else:
            print("  RUNS, BUT NOT AS EXPECTED -> compare 'got' with 'you should see'")
    return done


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check the Lab 1 tools without a model")
    parser.add_argument("--service", help="look up one service name (for the guided alias change)")
    parser.add_argument("--extension", action="store_true", help="check the optional sensor_trend tool")
    args = parser.parse_args(argv)
    config.ACTION_LOG = config.OUTPUT_DIR / "try_tools_action_log.csv"   # keep your real log clean

    if args.service:
        try:
            print(json.dumps(TOOL_FUNCTIONS["read_status"](service=args.service), indent=2, default=str))
        except NotImplementedError as exc:
            print(f"NOT YET -> {exc}")
        return
    if args.extension:
        if "sensor_trend" not in TOOL_FUNCTIONS:
            print("sensor_trend is not available: check src/extension.py for errors.")
            return
        done = run_checks(EXTENSION_CHECKS)
        print(f"\n{done} of {len(EXTENSION_CHECKS)} extension checks OK")
        return
    done = run_checks(CHECKS)
    print(f"\n{done} of {len(CHECKS)} checks OK" +
          ("  - all tools done. Next: run the agent." if done == len(CHECKS) else "  - fix the first one that is not OK."))


if __name__ == "__main__":
    main(sys.argv[1:])
