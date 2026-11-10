# Lab 1: Build a Tool-Using Agent

An enterprise service-request assistant. It classifies a request by
category and urgency, checks a simulated service-status table, searches a knowledge file and proposes a
next step, recording an action only after human approval. All data and actions are simulated.

Everything runs in Google Colab. Nothing is installed on your computer, and nothing costs money.

## Getting started

1. Unzip this folder and upload `lab1_service_agent` to **My Drive** in Google Drive.
2. In Drive, open `Lab1_Build_a_Tool_Using_Agent.ipynb` with Google Colab.
3. Runtime > Change runtime type > **T4 GPU**.
4. Add your free Gemini API key (from Google AI Studio) in **Colab Secrets** (key icon on the left) as
   `GEMINI_API_KEY`, and switch on notebook access. Never paste a key into a notebook or a file.
5. Run the two cells under *Start here*, then follow the notebook from top to bottom.

## What you write

Almost everything is already written. Seven short gaps remain to be completed,
plus a few settings. Each gap has a comment above it: what to write, why, a hint and how to check it.
More hints, one level at a time, are in `docs/HINTS.md`.

| File | What |
| --- | --- |
| `scripts/01_explore_data.py` | change the two EDIT settings (pandas filter, NumPy metric) |
| `src/tools.py` | CODE-ALONG 1 to 3 in `read_status`, 4 in `search_knowledge`, 5 in `record_proposed_action`; add one alias to `EXTRA_ALIASES` |
| `src/config.py` | switch on `STEP_LIMIT_ON`, `RESULT_VALIDATION_ON`, `APPROVAL_GATE_ON` |
| `src/safeguards.py` | CODE-ALONG 6 step limit, CODE-ALONG 7 approval answer |
| `docs/architecture.md` | Project 1 blueprint draft, capstone use-case brief, model comparison note |
| `src/extension.py` (optional) | write the `sensor_trend` tool from the hints |

To write a gap: open the file from the Files panel in Colab (double-click), replace the
`raise NotImplementedError(...)` line with your code at the same indentation, save with Ctrl+S, then
run the check (`try_tools.py` for gaps 1 to 5, `try_safeguards.py` for 6 and 7).

## Project map

```
lab1_service_agent/
  Lab1_Build_a_Tool_Using_Agent.ipynb   the notebook: one cell per step
  requirements.txt                      pinned packages (installed by the setup cell)
  data/                                 synthetic data (regenerate: %run scripts/generate_data.py)
  docs/architecture.md                  your Project 1 blueprint draft
  docs/HINTS.md                         more hints for every gap, one level at a time
  outputs/                              everything you produce: traces, logs, reports
  src/
    config.py          model names, paths, the three safeguard switches
    data_store.py      loads the data
    tool_schemas.py    tool contracts (JSON Schema) and the final-answer schema
    tools.py           the three tools                     <- CODE-ALONG 1 to 5
    safeguards.py      step limit, validation, approval    <- CODE-ALONG 6 and 7
    extension.py       optional fourth tool, sensor_trend  <- optional extension
    agent.py           the observe, reason, act, learn loop
    providers.py       open model (OpenAI library + Ollama), Gemini, mock
    state.py           what the agent remembers during a run
    trace.py           the execution trace
    evaluation.py      acceptance tests and comparison metrics
    mock_model.py      a rule-based stand-in (not an LLM) for when no model is available
  scripts/
    colab_setup.py     packages, Ollama, open model, data
    00_check_setup.py  every item should PASS
    01_explore_data.py pandas and NumPy step
    02_plain_vs_agent.py  the same request: plain model versus agent
    gpu_check.py       GPU confirmation for the tool passport
    try_tools.py       check your tools without a model (after every gap)
    try_safeguards.py  see what each safeguard stops
    03_run_agent.py    run the agent on one or more requests
    04_compare_models.py  open model versus Gemini on the benchmark
    05_run_tests.py    the four acceptance cases
    06_evidence_check.py  your evidence checklist
    generate_data.py   the synthetic dataset generator
```

## Providers

| `--provider` | What runs | When to use it |
| --- | --- | --- |
| `open` | an open model in Ollama, inside your Colab runtime, called through the OpenAI library | the default |
| `gemini` | Gemini through Google's `google-genai` library | the comparison step |
| `gemini_openai` | Gemini through the OpenAI library | no GPU available |
| `mock` | rule-based stand-in, not a model | no GPU and no API available |

Moving the `open` provider to OpenAI's paid models later only means changing `base_url`, the key and
the model name in `src/config.py` and `src/providers.py`.

## The four acceptance cases (target: 3 of 4)

| Case | Request | Passes when |
| --- | --- | --- |
| T1 status check | R01 email down for the finance team | `read_status("email")` in the trace and the answer uses its state |
| T2 knowledge search | R05 reset VPN token | `search_knowledge` returns KB-001, the answer cites it, no action logged |
| T3 invalid argument | R03 "Quantum Portal" | unknown service reported, no status invented |
| T4 approval gate | R02 restart payroll, reviewer says no | nothing written to the action log, the run stops cleanly |

The extra case R04 checks that a missing status is reported as unknown, never guessed.
