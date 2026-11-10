"""Colab setup for Lab 1. Run it first:

    %run scripts/colab_setup.py

It installs the Python packages, installs and starts Ollama inside this Colab
runtime, downloads the open model and generates the synthetic data if needed.
It takes a few minutes the first time in a runtime, and is safe to run again
(for example after Colab disconnects): finished steps are skipped.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _m in [m for m in list(sys.modules) if m == "src" or m.startswith("src.")]:
    del sys.modules[_m]
from src import config  # noqa: E402

OLLAMA_URL = "http://localhost:11434"


def step(msg):
    print(f"\n>>> {msg}", flush=True)


def run(cmd, quiet=True):
    out = subprocess.run(cmd, shell=True, capture_output=quiet, text=True)
    if out.returncode != 0:
        print(out.stdout[-2000:] if out.stdout else "", out.stderr[-2000:] if out.stderr else "")
        raise RuntimeError(f"Command failed: {cmd}")
    return out


def ollama_up() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/version", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def has_model(name: str) -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as r:
            models = [m["name"] for m in json.load(r).get("models", [])]
    except Exception:
        return False
    return any(m == name or m == f"{name}:latest" for m in models)


def pull_model(name: str) -> None:
    req = urllib.request.Request(f"{OLLAMA_URL}/api/pull", data=json.dumps({"model": name}).encode(),
                                 headers={"Content-Type": "application/json"})
    last_pct = -10
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            msg = json.loads(line)
            if "error" in msg:
                raise RuntimeError(f"Model download failed: {msg['error']}")
            total, done = msg.get("total"), msg.get("completed")
            if total and done:
                pct = int(100 * done / total)
                if pct >= last_pct + 10:
                    print(f"    {msg.get('status', 'downloading')}: {pct}% of {total / 1e9:.1f} GB", flush=True)
                    last_pct = pct
            elif msg.get("status") in ("success",):
                print("    download complete", flush=True)


def main():
    model = config.OPEN_MODEL
    started = time.time()

    step("1/4 Installing Python packages")
    run(f"{sys.executable} -m pip install -q -r {ROOT / 'requirements.txt'}")
    print("    done")

    step("2/4 Installing and starting Ollama in this runtime")
    if shutil.which("ollama") is None:
        run("apt-get -qq update && apt-get -qq install -y pciutils zstd > /dev/null")
        run("curl -fsSL https://ollama.com/install.sh | sh")
    if not ollama_up():
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        log = open(config.OUTPUT_DIR / "ollama.log", "a")
        env = dict(os.environ, OLLAMA_KEEP_ALIVE="60m")
        subprocess.Popen(["ollama", "serve"], stdout=log, stderr=log, env=env, start_new_session=True)
        for _ in range(30):
            if ollama_up():
                break
            time.sleep(1)
    if not ollama_up():
        raise RuntimeError("Ollama did not start. Check outputs/ollama.log, then rerun this cell.")
    print("    Ollama is running at", OLLAMA_URL)

    step(f"3/4 Downloading the open model {model}")
    if has_model(model):
        print("    already downloaded")
    else:
        pull_model(model)

    step("4/4 Synthetic data")
    if config.STATUS_CSV.exists() and config.REQUESTS_JSON.exists():
        print("    data/ already present")
    else:
        run(f"{sys.executable} {ROOT / 'scripts' / 'generate_data.py'}", quiet=False)
    config.TRACE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nSetup finished in {time.time() - started:.0f}s. Next: %run scripts/00_check_setup.py")


if __name__ == "__main__":
    main()
