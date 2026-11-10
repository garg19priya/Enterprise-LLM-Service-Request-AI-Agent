"""Central settings for Lab 1.

Change models here, or override them without editing code, for example:
    import os; os.environ["OPEN_MODEL"] = "llama3.2:3b"

API keys are NEVER stored in this file, in a notebook or anywhere in the
project folder. They are read from Colab Secrets (or an environment variable).
"""
import os
from pathlib import Path

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
KB_DIR = DATA_DIR / "knowledge_base"
STATUS_CSV = DATA_DIR / "service_status.csv"
READINGS_CSV = DATA_DIR / "sensor_readings.csv"
REQUESTS_JSON = DATA_DIR / "requests.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TRACE_DIR = OUTPUT_DIR / "traces"
# Read at call time by record_proposed_action, so scripts can redirect it.
ACTION_LOG = OUTPUT_DIR / "action_log.csv"

# ---------------------------------------------------------------- models
# Open model served by Ollama inside the Colab runtime (OpenAI-compatible API).
# qwen2.5:7b (about 4.7 GB) calls tools reliably on a free T4 GPU.
# llama3.2:3b (about 2 GB) is faster to download but less reliable with tools.
OPEN_MODEL = os.environ.get("OPEN_MODEL", "qwen2.5:7b")
OPEN_MODEL_BASE_URL = os.environ.get("OPEN_MODEL_BASE_URL", "http://localhost:11434/v1")

# Gemini model on the free tier of Google AI Studio. Flash-Lite has the largest free
# daily quota. Model names change over time: if the setup check fails, it lists the
# models your key can use; set os.environ["GEMINI_MODEL"] to one of them.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_KEY_NAME = "GEMINI_API_KEY"          # name of the Colab Secret
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
# The free tier allows only a few requests per minute; space calls out.
GEMINI_MIN_SECONDS_BETWEEN_CALLS = float(os.environ.get("GEMINI_MIN_SECONDS", "6"))

TEMPERATURE = 0.0
REQUEST_TIMEOUT_SECONDS = 180

# ---------------------------------------------------------------- safeguards
# Switch each safeguard on (False -> True), save, and rerun.
# Before you switch them on, see what happens without them; afterwards, test what
# each one stops. The code behind them is in src/safeguards.py (CODE-ALONG 6 and 7).
STEP_LIMIT_ON = False          # stop the agent after MAX_STEPS model turns
RESULT_VALIDATION_ON = False   # check every tool result against its contract
APPROVAL_GATE_ON = False       # a human must approve record_proposed_action

MAX_STEPS = 6                  # the step limit used when STEP_LIMIT_ON is True
COURSE_SAFETY_NET_STEPS = 15   # hard stop used by the course code even with the step limit off
MAX_FINAL_ANSWER_RETRIES = 1   # how often the agent may ask the model to repair invalid JSON

# Optional extension (after the main tasks): offer the sensor_trend tool to the
# model once you have written it in src/extension.py.
EXTENSION_TOOL_ON = False

CATEGORIES = ["access", "network", "application", "hardware", "facilities", "information"]
URGENCIES = ["low", "medium", "high", "critical"]


def get_secret(name: str):
    """Return a secret from Colab Secrets, falling back to an environment variable.

    In Colab: open the key icon in the left sidebar, add GEMINI_API_KEY and switch
    on 'Notebook access'. Returns None when the secret is missing.
    """
    try:
        from google.colab import userdata  # type: ignore
        value = userdata.get(name)
        if value:
            return value
    except Exception:
        pass
    return os.environ.get(name)
