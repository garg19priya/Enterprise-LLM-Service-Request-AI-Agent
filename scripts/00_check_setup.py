"""Setup check. Every item should show PASS before Lab 1 starts.

    %run scripts/00_check_setup.py
"""
import importlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _m in [m for m in list(sys.modules) if m == "src" or m.startswith("src.")]:
    del sys.modules[_m]
from src import config  # noqa: E402

RESULTS = []


def record(name, status, detail=""):
    RESULTS.append((name, status, detail))
    print(f"  [{status:<4}] {name}: {detail}", flush=True)


def check_python():
    v = sys.version_info
    record("Python", "PASS" if v >= (3, 10) else "FAIL", f"{v.major}.{v.minor}.{v.micro}")


def check_packages():
    for pkg, import_name in [("openai", "openai"), ("google-genai", "google.genai"), ("jsonschema", "jsonschema"),
                             ("numpy", "numpy"), ("pandas", "pandas")]:
        try:
            importlib.import_module(import_name)
            from importlib.metadata import version as installed_version
            version = installed_version(pkg)
            record(f"package {pkg}", "PASS", version)
        except Exception as exc:
            record(f"package {pkg}", "FAIL", f"{exc}. Run the setup cell again.")


def check_data():
    missing = [p.name for p in (config.STATUS_CSV, config.READINGS_CSV, config.REQUESTS_JSON) if not p.exists()]
    notes = len(list(config.KB_DIR.glob("KB-*.md"))) if config.KB_DIR.exists() else 0
    if missing or notes < 10:
        record("synthetic data", "FAIL", f"missing {missing or ''} notes={notes}. Run: %run scripts/generate_data.py")
    else:
        record("synthetic data", "PASS", "status table, readings, 10 notes and requests present")


def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            record("Colab GPU", "PASS", torch.cuda.get_device_name(0))
        else:
            record("Colab GPU", "WARN", "no GPU. Runtime > Change runtime type > T4 GPU, or use the Gemini fallback")
    except ImportError:
        record("Colab GPU", "WARN", "torch not installed (it is preinstalled on Colab)")


def check_open_model():
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as r:
            models = [m["name"] for m in json.load(r).get("models", [])]
        record("Ollama server", "PASS", "running")
    except Exception:
        record("Ollama server", "FAIL", "not running. Rerun: %run scripts/colab_setup.py")
        return
    if not any(m in (config.OPEN_MODEL, f"{config.OPEN_MODEL}:latest") for m in models):
        record("open model", "FAIL", f"{config.OPEN_MODEL} not downloaded (found: {models}). Rerun the setup cell.")
        return
    record("open model", "PASS", f"{config.OPEN_MODEL} downloaded")

    from src.providers import make_session
    try:
        session = make_session("open", "You are a test assistant. Use tools when asked.")
        session.add_user("Check the status of the email service.")
        turn = session.next_turn()
        if turn.tool_calls:
            record("open model tool call", "PASS",
                   f"asked for {turn.tool_calls[0].name} in {turn.latency_s:.1f}s (first call loads the model)")
        else:
            record("open model tool call", "WARN", f"answered without a tool: {turn.text[:80]!r}")
    except Exception as exc:
        record("open model tool call", "FAIL", str(exc)[:200])


def check_gemini():
    key = config.get_secret(config.GEMINI_KEY_NAME)
    if not key:
        record("Gemini API key", "FAIL", f"add {config.GEMINI_KEY_NAME} in Colab Secrets (key icon) with notebook access on")
        return
    record("Gemini API key", "PASS", "found in Colab Secrets (value not shown)")
    try:
        from google import genai
        client = genai.Client(api_key=key)
        from google.genai import types
        no_afc = types.GenerateContentConfig(
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        client.models.generate_content(model=config.GEMINI_MODEL, contents="Reply with the word OK.", config=no_afc)
        record("Gemini model", "PASS", f"{config.GEMINI_MODEL} answered a test call")
    except Exception as exc:
        detail = str(exc)[:160]
        try:
            names = [m.name.replace("models/", "") for m in client.models.list()
                     if "flash" in m.name and "generateContent" in (getattr(m, "supported_actions", None) or [])]
            detail += f" | models your key can use include: {names[:6]}. Set os.environ['GEMINI_MODEL']."
        except Exception:
            pass
        record("Gemini model", "FAIL", detail)


def main():
    RESULTS.clear()

    print("Lab 1 setup check\n")
    check_python()
    check_packages()
    check_data()
    check_gpu()
    check_open_model()
    check_gemini()

    fails = [r for r in RESULTS if r[1] == "FAIL"]
    warns = [r for r in RESULTS if r[1] == "WARN"]
    print(f"\n{len(RESULTS) - len(fails) - len(warns)} passed, {len(warns)} warnings, {len(fails)} failed")
    print("All required checks passed. You are ready for Lab 1." if not fails else
          "Fix the FAIL items above, then run this cell again.")


if __name__ == "__main__":
    main()
