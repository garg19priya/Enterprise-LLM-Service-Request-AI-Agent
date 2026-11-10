"""OPTIONAL EXTENSION, after the main tasks: write a fourth tool from scratch.

The tool: sensor_trend(service, hours=6)
    Is a service's sensor getting worse, better, or holding steady over the last
    few hours? read_status gives the mean and the maximum; this tool gives the
    DIRECTION, which helps the agent judge urgency (a temperature creeping up
    matters more than one that is high but falling).

Contract
    input    service: a service id or alias (same as read_status)
             hours:   how many of the most recent hourly readings to look at, 2 to 24 (default 6)
    returns  {"service", "metric", "hours", "older_mean", "newer_mean", "change", "threshold",
              "change_pct_of_threshold", "direction"}
    method   split the last `hours` readings into an older half and a newer half; change is
             newer_mean - older_mean; change_pct_of_threshold = 100 * change / threshold.
             direction is "rising" (> +5), "falling" (< -5) or "stable".
    why      comparing two averages smooths out the noise in single readings, and measuring
             the change against the alert threshold says how far the sensor moved towards an alarm.
    errors   {"error": "unknown_service", ...} exactly like read_status

Steps
    1. Write the function body below (about 10 lines). Check it without a model:
           %run scripts/try_tools.py --extension
    2. Tell the model about the tool: in src/config.py set EXTENSION_TOOL_ON = True.
       Its contract (EXTENSION_SPEC below) is then sent to the model with the other three.
    3. Run the agent on a request where the trend matters and look for sensor_trend in the trace:
           %run scripts/03_run_agent.py --provider open --request R08 --show-trace
    4. In docs/architecture.md, add the tool to the tools table.
"""
import numpy as np

from src.data_store import load_readings, load_status

EXTENSION_SPEC = {
    "name": "sensor_trend",
    "description": ("Show whether a service's sensor reading is rising, falling or stable over the last few "
                    "hours, with the first and last value and the percentage change. Use it to judge urgency."),
    "parameters": {
        "type": "object",
        "properties": {
            "service": {"type": "string", "description": "Service id or the name the user used, e.g. 'hvac'."},
            "hours": {"type": "integer", "minimum": 2, "maximum": 24,
                      "description": "How many recent hourly readings to compare (2 to 24). Default 6."},
        },
        "required": ["service"],
        "additionalProperties": False,
    },
}


def sensor_trend(service: str, hours: int = 6) -> dict:
    """See the module docstring for the contract."""
    # Hints, step by step (try each step yourself before reading the next):
    #
    # a) Find the service exactly as read_status does:
    #        from src.tools import _find_service_row
    #        status = load_status(); row = _find_service_row(status, service)
    #    If row is None, return the same unknown_service error dict as CODE-ALONG 1.
    #
    # b) Get this sensor's readings IN TIME ORDER: filter load_readings() where the
    #    "sensor_id" column equals row["sensor_id"], then .sort_values("timestamp").
    #
    # c) Keep only the last `hours` values as a NumPy array:
    #        values = <your rows>["value"].to_numpy(dtype=float)[-hours:]
    #    A negative slice [-6:] means "the last 6 items".
    #
    # d) Split the array in two halves: half = len(values) // 2 ;
    #    values[:half] is the older half, values[half:] the newer half.
    #    Take np.mean of each (as floats): older_mean, newer_mean.
    #
    # e) change = newer_mean - older_mean;  threshold = float(row["threshold"]);
    #    pct = 100 * change / threshold.
    #    direction: "rising" if pct > 5, "falling" if pct < -5, otherwise "stable".
    #
    # f) Return the dict described in the contract (use the key change_pct_of_threshold
    #    for pct). Round numbers for readability (round(x, 2)); hours is len(values) as an int.
    #
    # Expected (check with try_tools.py --extension):
    #    hvac over 6 hours  -> rising, about +15.9 % of threshold
    #    vpn  over 6 hours  -> stable, about -1.9 %
    #    erp  over 6 hours  -> falling, about -25 % (the service was drained for maintenance)
    raise NotImplementedError("EXTENSION (src/extension.py): write sensor_trend")
