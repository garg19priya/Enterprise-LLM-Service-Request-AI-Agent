"""The agent's three tools.

Complete the missing lines.

Most of each tool is already written. Five short gaps are marked
CODE-ALONG 1 to 5. Each gap tells you what to write, which variables are
already available and what the result should look like. Replace the
`raise NotImplementedError(...)` line in the gap with your code.

After every gap, save (Ctrl+S) and check your work:

    %run scripts/try_tools.py

Each tool follows its contract in src/tool_schemas.py:
  * fixed inputs, checked before the tool runs;
  * a dict as output, always;
  * a defined error dict when something is wrong, so the agent never has to guess.
"""
import csv
import re
from datetime import datetime

import numpy as np
import pandas as pd

from src import config
from src.data_store import load_knowledge_base, load_readings, load_status

STOPWORDS = {"a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "my", "i", "is", "it", "how",
             "do", "does", "can", "what", "where", "who", "why", "when", "me", "we", "our", "be", "are",
             "with", "this", "that", "please", "get", "has", "have", "not", "at", "from", "after", "too"}

# ----------------------------------------------------------------------------
# GUIDED CHANGE: teach the status tool a new name for a service.
# Users call services by many names. Left: what a user might type (lower case).
# Right: the service id it means (see data/service_status.csv, column "service").
# Add ONE line of your own, for example "teams call": "video", then check it:
#     %run scripts/try_tools.py --service "teams call"
# ----------------------------------------------------------------------------
EXTRA_ALIASES = {
    "exchange": "email",
    "outlook web": "email",
    "teams call": "video",
}


def _find_service_row(status: pd.DataFrame, name: str):
    """Match a service by id, by alias, or by an alias contained in the name the model passed.

    Returns the matching row of the status table, or None if nothing matches.
    """
    key = name.strip().lower()
    key = EXTRA_ALIASES.get(key, key)
    for _, row in status.iterrows():
        aliases = [a.strip() for a in str(row["aliases"]).split(";")]
        if key == row["service"] or key in aliases:
            return row
    best, best_len = None, 0
    for _, row in status.iterrows():
        for alias in [row["service"]] + [a.strip() for a in str(row["aliases"]).split(";")]:
            if re.search(rf"\b{re.escape(alias)}\b", key) and len(alias) > best_len:
                best, best_len = row, len(alias)
    return best


# ----------------------------------------------------------------------------
# Tool 1: read_status
# ----------------------------------------------------------------------------
def read_status(service: str) -> dict:
    """Return the current state of one service plus a NumPy summary of its sensor.

    Contract
      input   service: a service id or alias, e.g. "email"
      returns {"service", "display_name", "owner_team", "state", "last_incident",
               "last_updated", "metric", "sensor_mean", "sensor_max", "threshold",
               "above_threshold"}
      errors  {"error": "unknown_service", ...} when the service is not in the table;
              state is "unknown" (never guessed) when the status cell is empty.
    """
    status = load_status()                       # pandas DataFrame, one row per service
    row = _find_service_row(status, service)     # the matching row, or None

    # ------------------------------------------------------------------
    # CODE-ALONG 1: never invent a service.  (2 to 5 lines)
    # WHY  If `row` is None, the service does not exist. The tool must say
    #      so in the agreed way, so the agent reports it instead of guessing.
    # DO   If `row is None`, RETURN a dict with these four keys:
    #        "error":          the text "unknown_service"
    #        "service":        the name that was passed in (the variable `service`)
    #        "message":        a short sentence, e.g. f"'{service}' is not a known service."
    #        "known_services": the list of valid ids, sorted
    # HINT The valid ids are the column status["service"]; .tolist() turns a
    #      pandas column into a Python list, and sorted(...) sorts a list.
    # CHECK %run scripts/try_tools.py   ("Quantum Portal" should show unknown_service)
    # ------------------------------------------------------------------
    if row is None:
        return {"error": "unknown_service", "service": service,
                "message": f"'{service}' is not a known service.",
                "known_services": sorted(status["service"].tolist())}
    # ------------------------------------------------------------------
    # CODE-ALONG 2: never guess a missing status.  (1 or 2 lines)
    # WHY  One row (printing) has an EMPTY state cell on purpose. pandas reads
    #      an empty cell as NaN, a float, not a string.
    # DO   Create a variable `state`:
    #        * take row["state"]
    #        * if it is a non-empty string, keep it (strip any spaces)
    #        * otherwise use the text "unknown"
    # HINT isinstance(x, str) is True only for text, so it is False for NaN.
    #      A one-line form is:  value_if_true if condition else value_if_false
    # CHECK %run scripts/try_tools.py   (printing should show state "unknown")
    # ------------------------------------------------------------------
    state = row["state"]
    state = state.strip() if isinstance(state, str) and state.strip() else "unknown"
    # The sensor readings for this service, as a NumPy array of floats.
    readings = load_readings()
    values = readings.loc[readings["sensor_id"] == row["sensor_id"], "value"].to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # CODE-ALONG 3: summarise the sensor with NumPy.  (2 lines)
    # WHY  The agent needs a number, not 24 raw readings, to judge a service.
    # DO   Create two variables from the array `values`:
    #        mean = the average of the readings, as a Python float
    #        peak = the largest reading, as a Python float
    # HINT np.mean(...) and np.max(...) each take an array and return one number;
    #      wrap it in float(...). (Every service has readings, so `values` is never
    #      empty here.)
    # CHECK %run scripts/try_tools.py   (email: sensor_max should be 2080.0)
    #      Until then the placeholder below keeps the tool running with no numbers.
    # ------------------------------------------------------------------
    # mean = peak = None   # <- CODE-ALONG 3: replace this placeholder line with your two lines
    mean = float(np.mean(values))
    peak = float(np.max(values))

    threshold = float(row["threshold"])
    return {
        "service": row["service"],
        "display_name": row["display_name"],
        "owner_team": row["owner_team"],
        "state": state,
        "last_incident": row["last_incident"],
        "last_updated": row["last_updated"],
        "metric": row["metric"],
        "sensor_mean": round(mean, 2) if mean is not None else None,
        "sensor_max": round(peak, 2) if peak is not None else None,
        "threshold": threshold,
        "above_threshold": bool(peak is not None and peak > threshold),
    }


# ----------------------------------------------------------------------------
# Tool 2: search_knowledge
# ----------------------------------------------------------------------------
def _tokens(text: str) -> list:
    """Lower-case words of a text, without very common words. "Reset my VPN!" -> ["reset", "vpn"]"""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS and len(t) > 1]


def search_knowledge(query: str, top_k: int = 3) -> dict:
    """Keyword search over the knowledge base.

    Contract
      input   query: a few keywords; top_k: 1 to 5 (default 3)
      returns {"query": ..., "results": [{"note_id", "title", "snippet", "score"}, ...]}
      errors  an empty results list when nothing matches (never an invented note)
    """
    words = _tokens(query)                       # e.g. ["reset", "vpn", "token"]
    results = []
    for note in load_knowledge_base():
        title_words = set(_tokens(note["title"]))                     # words in the note's title
        keyword_words = set(_tokens(" ".join(note["keywords"])))      # words in its keyword line
        body_words = _tokens(note["body"])                            # every word of its text, in order

        # ------------------------------------------------------------------
        # CODE-ALONG 4: score how well this note matches the query.  (1 to 4 lines)
        # WHY  The agent should get the most relevant notes first.
        # DO   Create a number `score`. For EACH query word w in `words`, add:
        #        3 if w is in title_words
        #        2 if w is in keyword_words
        #        the number of times w appears in body_words, but at most 2
        # HINT `w in title_words` is True or False, and Python counts True as 1,
        #      so 3 * (w in title_words) is 3 or 0.  body_words.count(w) counts
        #      a word; min(count, 2) caps it.  Start with score = 0 and add inside
        #      a for loop, or use sum(... for w in words) on one line.
        # CHECK %run scripts/try_tools.py   ("reset vpn token" should rank KB-001 first)
        # ------------------------------------------------------------------
        # raise NotImplementedError("CODE-ALONG 4 (src/tools.py): compute the note's score")
        score = 0
        for w in words:
            score += 3 * (w in title_words) + 2 * (w in keyword_words) + min(body_words.count(w), 2)
        if score > 0:
            snippet = " ".join(note["body"].split())[:220]
            results.append({"note_id": note["note_id"], "title": note["title"],
                            "snippet": snippet, "score": int(score)})
    results.sort(key=lambda r: (-r["score"], r["note_id"]))         # best first, ties by id
    return {"query": query, "results": results[:top_k]}


# ----------------------------------------------------------------------------
# Tool 3: record_proposed_action
# ----------------------------------------------------------------------------
ACTION_LOG_FIELDS = ["action_id", "recorded_at", "request_id", "action", "reason", "approved_by"]


def count_logged_actions() -> int:
    path = config.ACTION_LOG
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)


def record_proposed_action(request_id: str, action: str, reason: str, *, approved_by=None) -> dict:
    """Append an approved action to outputs/action_log.csv (all actions are simulated).

    Contract
      input   request_id, action, reason  (approved_by is set by the agent's approval gate,
              never by the model)
      returns {"action_id", "recorded_at", "request_id", "action"}
      errors  {"error": "approval_required"} when no human approved the action
    """
    # ------------------------------------------------------------------
    # CODE-ALONG 5: no approval, no action.  (2 or 3 lines)
    # WHY  This is the only tool that changes something. It must refuse to
    #      write unless a human approved, even if the agent's code has a bug.
    # DO   If `approved_by` is empty (None or ""), RETURN a dict with:
    #        "error":   the text "approval_required"
    #        "message": a short sentence saying human approval is needed
    # HINT `if not approved_by:` is True for both None and "".
    # CHECK %run scripts/try_tools.py   (the action should show approval_required)
    # ------------------------------------------------------------------
    # raise NotImplementedError("CODE-ALONG 5 (src/tools.py): refuse when approved_by is empty")
    if not approved_by:
        return {"error": "approval_required",
                "message": "This action needs human approval before it can be recorded."}
    path = config.ACTION_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    action_id = f"ACT-{count_logged_actions() + 1:04d}"
    recorded_at = datetime.now().isoformat(timespec="seconds")
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ACTION_LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({"action_id": action_id, "recorded_at": recorded_at, "request_id": request_id,
                         "action": action, "reason": reason, "approved_by": approved_by})
    return {"action_id": action_id, "recorded_at": recorded_at, "request_id": request_id, "action": action}


# Registry used by the agent: tool name -> Python function
TOOL_FUNCTIONS = {
    "read_status": read_status,
    "search_knowledge": search_knowledge,
    "record_proposed_action": record_proposed_action,
}

try:  # the optional extension tool, once it has been written (see src/extension.py)
    from src.extension import sensor_trend
    TOOL_FUNCTIONS["sensor_trend"] = sensor_trend
except ImportError:
    pass
