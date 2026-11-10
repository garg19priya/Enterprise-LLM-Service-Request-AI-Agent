"""Run the agent on one or more requests and (optionally) show the trace.

    %run scripts/03_run_agent.py --provider open --request R01 --show-trace
    %run scripts/03_run_agent.py --provider open --request R03 R04       # guided debugging
    %run scripts/03_run_agent.py --provider open --request R02           # approval prompt
    %run scripts/03_run_agent.py --provider open --text "The Wi-Fi on floor 2 is down"

Providers: open (Ollama in Colab), gemini, gemini_openai (no-GPU fallback), mock (no model).
Approval: interactive (you answer y/n), approve, deny.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Reload the project code on every run, so edits to src/*.py are picked up.
for _m in [m for m in list(sys.modules) if m == "src" or m.startswith("src.")]:
    del sys.modules[_m]

from src import config  # noqa: E402
from src.agent import run_agent  # noqa: E402
from src.trace import show_trace  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the Lab 1 agent")
    parser.add_argument("--provider", default="open", choices=["open", "gemini", "gemini_openai", "mock"])
    parser.add_argument("--request", nargs="*", default=["R01"], help="request ids, e.g. R01 R03")
    parser.add_argument("--text", help="run on your own request text instead")
    parser.add_argument("--approval", default="interactive", choices=["interactive", "approve", "deny"])
    parser.add_argument("--max-steps", type=int, default=config.MAX_STEPS)
    parser.add_argument("--show-trace", action="store_true")
    args = parser.parse_args(argv)

    targets = [args.text] if args.text else args.request
    for target in targets:
        result = run_agent(target, provider=args.provider, approval_mode=args.approval, max_steps=args.max_steps)
        if args.show_trace:
            print("\nTRACE")
            show_trace(result.trace_path)


if __name__ == "__main__":
    main(sys.argv[1:])
