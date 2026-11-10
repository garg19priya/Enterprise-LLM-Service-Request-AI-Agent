"""Inspect the data the status tool will use.

    %run scripts/01_explore_data.py

pandas: load and filter the service-status table, find the missing value.
NumPy:  calculate the mean and maximum of each sensor's readings.

YOUR CHANGE: edit the two settings marked EDIT below, predict what will
change in the output, then run the cell again. Output: outputs/data_summary.txt
"""
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _m in [m for m in list(sys.modules) if m == "src" or m.startswith("src.")]:
    del sys.modules[_m]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.data_store import load_readings, load_status  # noqa: E402

# EDIT: which states count as "needs attention"? Try adding "maintenance".
STATES_TO_FLAG = ["degraded", "outage"]
# EDIT: which metric do you want to look at closely? e.g. "temperature_c", "latency_ms", "packet_loss_pct"
METRIC_OF_INTEREST = "latency_ms"


def main():
    lines = []

    def out(text=""):
        print(text)
        lines.append(text)

    status = load_status()
    readings = load_readings()

    out("SERVICE STATUS TABLE (pandas)")
    out(status[["service", "display_name", "owner_team", "state", "last_incident"]].to_string(index=False))

    out(f"\nServices needing attention (state in {STATES_TO_FLAG}):")
    flagged = status[status["state"].isin(STATES_TO_FLAG)]
    out(flagged[["service", "state", "owner_team"]].to_string(index=False) if len(flagged) else "  none")

    missing = status[status["state"].isna()]
    out(f"\nRows with a MISSING state: {missing['service'].tolist()}  <- the agent must say 'unknown', never guess")

    out("\nSENSOR SUMMARY (NumPy)")
    out(f"{'service':<12}{'metric':<17}{'mean':>10}{'max':>10}{'threshold':>11}  above?")
    for _, row in status.iterrows():
        values = readings.loc[readings["sensor_id"] == row["sensor_id"], "value"].to_numpy(dtype=float)
        mean, peak = np.mean(values), np.max(values)
        above = "YES" if peak > row["threshold"] else ""
        out(f"{row['service']:<12}{row['metric']:<17}{mean:>10.2f}{peak:>10.2f}{row['threshold']:>11}  {above}")

    focus = readings[readings["metric"] == METRIC_OF_INTEREST]
    if len(focus):
        values = focus["value"].to_numpy(dtype=float)
        out(f"\nAcross all '{METRIC_OF_INTEREST}' sensors: mean {np.mean(values):.2f}, max {np.max(values):.2f}, "
            f"standard deviation {np.std(values):.2f}, {len(values)} readings")
        last = focus.sort_values("timestamp").groupby("service").tail(1)[["service", "timestamp", "value"]]
        out("Latest reading per service:\n" + last.to_string(index=False))
    else:
        out(f"\nNo readings with metric '{METRIC_OF_INTEREST}'. Options: {sorted(readings['metric'].unique())}")

    out(f"\nSettings used: STATES_TO_FLAG={STATES_TO_FLAG}, METRIC_OF_INTEREST={METRIC_OF_INTEREST!r}")
    out(f"Generated {datetime.now().isoformat(timespec='seconds')}")

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.OUTPUT_DIR / "data_summary.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved {path}")


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    main()
