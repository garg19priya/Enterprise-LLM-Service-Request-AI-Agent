"""Check your Lab 1 evidence.

    %run scripts/06_evidence_check.py

Lists the Lab 1 evidence items and writes outputs/evidence_checklist.md.
Everything stays in your Google Drive project folder; nothing else to upload.
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def gap_status():
    """Behavioural check of the CODE-ALONG gaps and switches (not a text search)."""
    sys.path.insert(0, str(ROOT))
    for name in [m for m in list(sys.modules) if m == "src" or m.startswith("src.")]:
        del sys.modules[name]
    try:
        from src.evaluation import code_along_status
        return code_along_status()
    except Exception:
        return {}


def has_trace(provider):
    for path in (OUT / "traces").glob(f"*_{provider}_*.jsonl"):
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if any(e["event"] == "tool_result" and "error" not in (e.get("result") or {}) for e in events):
            return path.name
    return None


def main():
    items = []
    gaps = gap_status()

    def item(ok, what, passport, hint):
        items.append((ok, what, passport, hint))

    data = OUT / "data_summary.txt"
    item(data.exists(), "outputs/data_summary.txt (pandas filter and NumPy statistics)", "pandas, NumPy",
         "%run scripts/01_explore_data.py")
    gpu = OUT / "gpu_check.txt"
    item(gpu.exists() and "RESULT: PASS" in gpu.read_text(encoding="utf-8"),
         "outputs/gpu_check.txt with GPU device confirmation", "Google Colab with GPU", "%run scripts/gpu_check.py")
    item(all(gaps.get(k) for k in ("CA1", "CA2", "CA3", "CA4", "CA5")),
         "src/tools.py: CODE-ALONG 1 to 5 work", "Python", "%run scripts/try_tools.py and finish the gaps it lists")
    open_trace = has_trace("open")
    item(bool(open_trace), f"a trace with a successful tool call via the open model ({open_trace or 'none yet'})",
         "OpenAI API (library) + Ollama", "%run scripts/03_run_agent.py --provider open --request R01")
    item(bool(has_trace("gemini") or has_trace("gemini_openai")), "a Gemini trace", "Gemini API",
         "%run scripts/04_compare_models.py")
    comp = OUT / "comparison.csv"
    item(comp.exists(), "outputs/comparison.csv and comparison_summary.md", "Gemini API",
         "%run scripts/04_compare_models.py")
    item(bool(gaps.get("switches_on") and gaps.get("CA6") and gaps.get("CA7")), "safeguards: three switches on in src/config.py, CODE-ALONG 6 and 7 done", "Python",
         "switch them on in src/config.py; finish CODE-ALONG 6 and 7 in src/safeguards.py")
    report = OUT / "test_report.md"
    passed = 0
    if report.exists():
        m = re.search(r"(\d) of 4 acceptance cases passed", report.read_text(encoding="utf-8"))
        passed = int(m.group(1)) if m else 0
    item(passed >= 3, f"outputs/test_report.md with at least 3 of 4 passing (now {passed})", "-",
         "%run scripts/05_run_tests.py --provider open")
    arch = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    item("TODO" not in arch, "docs/architecture.md: blueprint, use-case brief and model comparison note filled in",
         "Project 1", "replace every TODO in docs/architecture.md")

    done = sum(i[0] for i in items)
    lines = [f"# Lab 1 evidence checklist ({datetime.now().isoformat(timespec='seconds')})", "",
             f"{done} of {len(items)} items complete", "", "| Done | Evidence | Passport tool | How to finish |",
             "| --- | --- | --- | --- |"]
    lines += [f"| {'yes' if ok else 'NO'} | {what} | {pp} | {'' if ok else hint} |" for ok, what, pp, hint in items]
    (OUT / "evidence_checklist.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for ok, what, _pp, hint in items:
        print(f"  [{'x' if ok else ' '}] {what}" + ("" if ok else f"\n        -> {hint}"))
    print(f"\n{done} of {len(items)} complete. Saved outputs/evidence_checklist.md")


if __name__ == "__main__":
    main()
