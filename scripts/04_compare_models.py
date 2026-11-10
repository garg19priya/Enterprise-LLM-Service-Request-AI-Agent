"""Run the same benchmark through two providers and compare.

    %run scripts/04_compare_models.py                              # open model vs Gemini
    %run scripts/04_compare_models.py --providers gemini_openai gemini   # no-GPU fallback

Writes outputs/comparison.csv (one row per request and provider) and
outputs/comparison_summary.md. Actions are auto-approved and written to a
separate log (outputs/benchmark_action_log.csv) so your own log stays clean.

YOUR TASK: read both outputs and write ONE observed behavioural difference
in docs/architecture.md, section "Model comparison note".
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Reload the project code on every run, so edits to src/*.py are picked up.
for _m in [m for m in list(sys.modules) if m == "src" or m.startswith("src.")]:
    del sys.modules[_m]

import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.agent import run_agent  # noqa: E402
from src.data_store import load_requests  # noqa: E402
from src.evaluation import comparison_row  # noqa: E402

METRICS = ["final_answer_valid", "expected_tools_used", "evidence_grounded", "category_correct", "urgency_correct"]


def md_table(df) -> str:
    head = "| " + " | ".join(df.columns) + " |"
    rule = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, rule] + rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare providers on the benchmark")
    parser.add_argument("--providers", nargs="+", default=["open", "gemini"])
    parser.add_argument("--requests", nargs="*", help="request ids (default: the 8 benchmark requests)")
    args = parser.parse_args(argv)

    requests = load_requests()
    chosen = [r for r in requests if (r["id"] in args.requests if args.requests else r["benchmark"])]
    config.ACTION_LOG = config.OUTPUT_DIR / "benchmark_action_log.csv"

    rows = []
    for provider in args.providers:
        print(f"\n######## {provider}: {len(chosen)} requests ########")
        for req in chosen:
            result = run_agent(req, provider=provider, approval_mode="approve", verbose=False)
            row = comparison_row(req, result)
            rows.append(row)
            print(f"  {req['id']}: stop={row['stop_reason']:<20} valid={row['final_answer_valid']!s:<5} "
                  f"tools={row['tools_called'] or '-':<40} {row['latency_s']}s")
            if result.error:
                print(f"       error: {result.error[:300]}")

    df = pd.DataFrame(rows)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.OUTPUT_DIR / "comparison.csv", index=False)

    summary = df.groupby(["provider", "model"]).agg(
        requests=("request_id", "count"),
        **{m: (m, "mean") for m in METRICS},
        invalid_tool_arguments=("invalid_tool_arguments", "sum"),
        mean_steps=("steps", "mean"),
        mean_latency_s=("latency_s", "mean"),
    ).reset_index()
    for m in METRICS:
        summary[m] = (summary[m] * 100).round(0).astype(int).astype(str) + "%"
    summary["mean_steps"] = summary["mean_steps"].round(1)
    summary["mean_latency_s"] = summary["mean_latency_s"].round(1)

    md = ["# Model comparison", "",
          "Same benchmark requests, same tools, same prompt, temperature 0.", "",
          md_table(summary), "",
          "Per-request results: outputs/comparison.csv", "",
          "Structured-output validity = share of runs that ended with a final answer matching the schema.",
          "Evidence grounded = every item the answer cites came from a tool result in that run."]
    (config.OUTPUT_DIR / "comparison_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + summary.to_string(index=False))
    print("\nSaved outputs/comparison.csv and outputs/comparison_summary.md")
    print("Now write one behavioural difference in docs/architecture.md (Model comparison note).")


if __name__ == "__main__":
    main(sys.argv[1:])
