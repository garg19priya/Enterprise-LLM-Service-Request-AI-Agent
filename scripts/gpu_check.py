"""Confirm the Colab GPU (tool-passport checkpoint).

    %run scripts/gpu_check.py --size 2048

YOUR CHANGE: before running, write down whether you expect the GPU or the CPU
to be faster for this matrix size. Then try --size 256 and --size 4096.
Which is faster at each size, and why?

Saves outputs/gpu_check.txt. A CPU-only run is recorded as FAIL: it does not
count for the passport.
"""
import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "gpu_check.txt"


def timed_matmul(torch, device, size, repeats=5):
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    (a @ b).sum().item()                       # warm-up
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        c = a @ b
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats, float(c[0, 0])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Colab GPU check")
    parser.add_argument("--size", type=int, default=2048, help="matrix size N for an N x N multiplication")
    args = parser.parse_args(argv)

    lines = [f"Lab 1 GPU check  {datetime.now().isoformat(timespec='seconds')}", f"matrix size: {args.size}"]
    try:
        import torch
    except ImportError:
        lines.append("RESULT: FAIL - torch is not installed (it is preinstalled on Colab)")
        torch = None

    if torch is not None:
        cpu_s, _ = timed_matmul(torch, "cpu", args.size)
        lines.append(f"CPU time per multiplication: {cpu_s * 1000:.1f} ms")
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            gpu_s, value = timed_matmul(torch, "cuda", args.size)
            lines += [f"GPU device: {name}",
                      f"GPU time per multiplication: {gpu_s * 1000:.1f} ms",
                      f"GPU speed-up: {cpu_s / gpu_s:.1f}x",
                      f"tensor device confirmed: cuda:0 (sample value {value:.4f})",
                      "RESULT: PASS - GPU-backed computation confirmed"]
        else:
            lines.append("RESULT: FAIL - no GPU in this runtime. Runtime > Change runtime type > T4 GPU, then rerun."
                         " A CPU run does not count for the passport.")

    if shutil.which("nvidia-smi"):
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv"],
                             capture_output=True, text=True).stdout.strip()
        lines += ["", "nvidia-smi:", smi]
    if shutil.which("ollama"):
        ps = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout.strip()
        lines += ["", "ollama ps (PROCESSOR should show GPU once the model has been used):", ps or "(no model loaded yet)"]

    text = "\n".join(lines)
    print(text)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text + "\n", encoding="utf-8")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:])
