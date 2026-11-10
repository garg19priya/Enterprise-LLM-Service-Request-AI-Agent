# Lab 1 hints

Use these when the comment next to a gap is not enough. Read **one level at a time**:
try again after each level before reading the next. There are no full answers here.

After every gap: save the file (Ctrl+S), then run `%run scripts/try_tools.py`
(gaps 1 to 5) or `%run scripts/try_safeguards.py` (gaps 6 and 7).

---

## Before you start: four Python ideas used in every gap

| Idea | Example | Result |
| --- | --- | --- |
| A dict is written with `{ "key": value }` | `{"error": "unknown_service", "service": "x"}` | a dict with two keys |
| `return` ends the function and hands back a value | `return {"error": "x"}` | nothing after it runs |
| A comparison is already True or False | `3 < 6` | `True` |
| A one-line choice | `"big" if n > 10 else "small"` | `"small"` when n is 4 |

Indentation matters: your line must line up with the code around it (4 spaces per level).

---

## CODE-ALONG 1: never invent a service (`src/tools.py`, `read_status`)

**Level 1 (what).** The line just above the gap set `row`. When the service is unknown,
`row` is `None`. Write an `if` that checks for that, and inside it `return` a dict.

**Level 2 (how).** The check is `if row is None:`. The dict needs four keys:
`"error"`, `"service"`, `"message"` and `"known_services"`. The first is the fixed
text `"unknown_service"`; the second is the variable `service` (the name passed in).

**Level 3 (the tricky part).** For the list of valid ids: `status["service"]` is a pandas
column. `.tolist()` turns it into a Python list, and `sorted(...)` puts it in order.
For the message, an f-string puts a variable inside text: `f"'{service}' is not a known service."`

**You know it works when** `try_tools.py` shows `OK` for `read_status({"service": "Quantum Portal"})`.

---

## CODE-ALONG 2: never guess a missing status (`src/tools.py`, `read_status`)

**Level 1 (what).** Make a variable called `state`. It should be the text in `row["state"]`,
or `"unknown"` when that cell is empty.

**Level 2 (why it is tricky).** pandas reads an empty CSV cell as `NaN` ("not a number"),
which is a float, not a string. So `if row["state"] == "":` does not work.
`isinstance(x, str)` asks "is x text?" and is False for `NaN`.

**Level 3 (shape).** Store `row["state"]` in a variable first, then use a one-line choice:
`<keep it, stripped> if <it is a string and not blank> else "unknown"`.
`.strip()` removes spaces; a blank string counts as False in a condition.

**You know it works when** `printing` shows `"state": "unknown"`.

---

## CODE-ALONG 3: summarise the sensor with NumPy (`src/tools.py`, `read_status`)

**Level 1 (what).** Two variables: `mean` (the average of `values`) and `peak` (the largest).

**Level 2 (how).** `values` is already a NumPy array. NumPy has a function for each:
`np.mean(...)` and `np.max(...)`.

**Level 3 (detail).** Wrap each in `float(...)` so the result is a plain Python number
(the JSON trace cannot store NumPy's own number type).

**You know it works when** `email` shows `"sensor_max": 2080.0` and `"above_threshold": true`.

---

## CODE-ALONG 4: score a knowledge note (`src/tools.py`, `search_knowledge`)

**Level 1 (what).** One number, `score`, for the current `note`. Each query word that
appears in the title is worth 3, in the keywords 2, and in the body 1 per appearance (at most 2).

**Level 2 (the trick).** `w in title_words` is True or False, and Python treats True as 1.
So `3 * (w in title_words)` is 3 when the word is in the title and 0 when it is not.
`body_words.count(w)` counts appearances; `min(..., 2)` caps it at 2.

**Level 3 (shape).** Two ways, both fine:

- a loop: start `score = 0`, then `for w in words:` and add the three parts with `score += ...`
- one line: `score = sum( <the three parts added together> for w in words )`

**You know it works when** `"reset vpn token"` puts `KB-001` first and `"zebra migration"` returns `[]`.

---

## CODE-ALONG 5: no approval, no action (`src/tools.py`, `record_proposed_action`)

**Level 1 (what).** At the top of the function: if nobody approved, return an error
instead of writing to the log.

**Level 2 (how).** `approved_by` is `None` or `""` when nobody approved. `if not approved_by:`
is True for both.

**Level 3 (shape).** Inside the `if`, return a dict with `"error": "approval_required"` and a
`"message"` explaining that human approval is needed.

**You know it works when** the last check in `try_tools.py` shows `approval_required`.

---

## Guided change: add a service alias (`src/tools.py`, `EXTRA_ALIASES`)

Add one line inside the `EXTRA_ALIASES` dict, in the same style as the two lines already there:
`"<what a user might type, lower case>": "<service id>",`. Service ids are in
`data/service_status.csv`, column `service`. Don't forget the comma at the end of the line.
Check it: `%run scripts/try_tools.py --service "your words"`.

---

## Switching on the safeguards (`src/config.py`)

Change `False` to `True` for `STEP_LIMIT_ON`, `RESULT_VALIDATION_ON` and `APPROVAL_GATE_ON`.
Capital T: `true` is not Python. Save, then rerun `try_safeguards.py`.

---

## CODE-ALONG 6: the step limit (`src/safeguards.py`, `within_step_limit`)

**Level 1 (what).** Answer one question with True or False: may the agent take another step?

**Level 2 (how).** It may while `steps_taken` is smaller than `max_steps`.

**Level 3 (shape).** A comparison is already True or False, so the whole function body is
one `return` of that comparison. Check the edge: with `max_steps = 6`, step 6 must be refused.

**You know it works when** `try_safeguards.py` says the agent STOPPED after 1 step.

---

## CODE-ALONG 7: approve only on a clear yes (`src/safeguards.py`, `request_approval`)

**Level 1 (what).** Make a variable `approved` that is True only for a clear yes.

**Level 2 (how).** `answer` is already lower case with spaces removed. A clear yes is
`"y"` or `"yes"`, nothing else.

**Level 3 (shape).** `x in ("a", "b")` is True when `x` equals either value.

**You know it works when** `try_safeguards.py` shows `maybe` and an empty answer DENIED.

---

## Optional extension: `sensor_trend` (`src/extension.py`)

The steps a) to f) in the file are the hints. Two more:

- `df.sort_values("timestamp")` returns the rows in time order; filter first, then sort.
- `array[-6:]` is the last six items. With `half = len(values) // 2` (`//` divides and drops
  the remainder), `values[:half]` is the older half and `values[half:]` the newer half.
- Why two averages and not first versus last? One noisy reading can swing a single value;
  averages of three readings each are much steadier.

Check it with `%run scripts/try_tools.py --extension`.
