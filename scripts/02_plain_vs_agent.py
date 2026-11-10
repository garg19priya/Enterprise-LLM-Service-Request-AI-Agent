"""The same request answered two ways: plain model versus agent.

    %run scripts/02_plain_vs_agent.py --provider open --request R01

Run it once the three tools work (try_tools.py shows 7 of 7).

1. Plain model: no tools, no state, no checks. Watch it guess.
2. Agent: chooses a tool, reads the real (simulated) status, cites its evidence and stops.
"""
import argparse
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Reload the project code on every run, so edits to src/*.py are picked up.
for _m in [m for m in list(sys.modules) if m == "src" or m.startswith("src.")]:
    del sys.modules[_m]

from src.agent import run_agent, run_plain  # noqa: E402
from src.data_store import get_request  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plain model versus agent")
    parser.add_argument("--provider", default="open", choices=["open", "gemini", "gemini_openai", "mock"])
    parser.add_argument("--request", default="R01")
    args = parser.parse_args(argv)

    req = get_request(args.request)
    print(f"REQUEST {req['id']}: {req['text']}\n")
    print("1) PLAIN MODEL (no tools)")
    print("-" * 70)
    print(textwrap.fill(run_plain(req, args.provider), 100))
    print("\nWhat does it know about the real status? Where did that come from?\n")
    print("2) AGENT (tools, trace, checks)")
    print("-" * 70)
    run_agent(req, provider=args.provider)


if __name__ == "__main__":
    main(sys.argv[1:])
