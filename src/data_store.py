"""Loads the synthetic course data. No changes are needed in this file for Lab 1."""
import json
import re
from functools import lru_cache

import pandas as pd

from src import config


def _require(path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found. Run the setup cell, or: %run scripts/generate_data.py")


@lru_cache(maxsize=1)
def load_status() -> pd.DataFrame:
    """Service-status table. Missing states stay as NaN on purpose (see the printing row)."""
    _require(config.STATUS_CSV)
    return pd.read_csv(config.STATUS_CSV)


@lru_cache(maxsize=1)
def load_readings() -> pd.DataFrame:
    """Hourly sensor readings for every service."""
    _require(config.READINGS_CSV)
    return pd.read_csv(config.READINGS_CSV, parse_dates=["timestamp"])


@lru_cache(maxsize=1)
def load_knowledge_base() -> list:
    """Knowledge notes as dicts: note_id, title, keywords, body."""
    if not config.KB_DIR.exists():
        raise FileNotFoundError("data/knowledge_base not found. Run: %run scripts/generate_data.py")
    notes = []
    for path in sorted(config.KB_DIR.glob("KB-*.md")):
        text = path.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip()
        match = re.search(r"^Keywords:\s*(.+)$", text, flags=re.MULTILINE)
        keywords = [k.strip().lower() for k in match.group(1).split(",")] if match else []
        body = text.split("\n\n", 2)[-1].strip()
        notes.append({"note_id": path.stem, "title": title, "keywords": keywords, "body": body})
    return notes


def load_requests() -> list:
    _require(config.REQUESTS_JSON)
    return json.loads(config.REQUESTS_JSON.read_text(encoding="utf-8"))


def get_request(request_id: str) -> dict:
    for req in load_requests():
        if req["id"].lower() == request_id.lower():
            return req
    raise KeyError(f"No request with id {request_id}. Valid ids: {[r['id'] for r in load_requests()]}")

