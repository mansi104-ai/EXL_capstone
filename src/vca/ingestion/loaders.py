"""Loaders that turn an uploaded conversation into a normalized dict.

Supported formats:
  * JSON  — {"conversation_id":..., "turns":[{"speaker":..,"text":..}, ...]}
            or a bare list of such turns.
  * CSV   — columns: speaker,text  (header optional).
  * TXT   — one line per turn, "Speaker: text". Lines without a recognised
            speaker prefix default to the customer.

The normalized dict matches the sample transcripts:
  {"conversation_id": str, "turns": [{"speaker": "customer"|"handler", "text": str}]}
"""
from __future__ import annotations

import csv
import io
import json
import re
from typing import Any

from ..schemas import Speaker

_HANDLER_WORDS = {"handler", "agent", "advisor", "adviser", "rep", "staff", "you", "operator"}
_CUSTOMER_WORDS = {"customer", "caller", "client", "user", "them", "member"}
_LINE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z _-]{0,20})\s*[:>-]\s*(.*)$")


def _norm_speaker(raw: str | None) -> Speaker:
    if raw:
        low = raw.strip().lower()
        if low in _HANDLER_WORDS:
            return Speaker.HANDLER
        if low in _CUSTOMER_WORDS:
            return Speaker.CUSTOMER
    # Default unknown speakers to the customer (they are the ones assessed).
    return Speaker.CUSTOMER


def _clean_turns(raw_turns: list[dict[str, Any]]) -> list[dict[str, str]]:
    turns = []
    for t in raw_turns:
        text = str(t.get("text", "")).strip()
        if not text:
            continue
        turns.append({"speaker": _norm_speaker(t.get("speaker")).value, "text": text})
    return turns


def load_json(content: str) -> dict:
    data = json.loads(content)
    if isinstance(data, list):
        return {"conversation_id": "UPLOAD", "turns": _clean_turns(data)}
    conv_id = str(data.get("conversation_id", "UPLOAD"))
    return {"conversation_id": conv_id, "turns": _clean_turns(data.get("turns", []))}


def load_csv(content: str) -> dict:
    reader = csv.reader(io.StringIO(content))
    rows = [r for r in reader if any(cell.strip() for cell in r)]
    if not rows:
        return {"conversation_id": "UPLOAD", "turns": []}
    # Detect a header row.
    start = 0
    header = [c.strip().lower() for c in rows[0]]
    s_idx, t_idx = 0, 1
    if "text" in header:
        start = 1
        t_idx = header.index("text")
        s_idx = header.index("speaker") if "speaker" in header else (1 - t_idx if len(header) > 1 else 0)
    raw = []
    for r in rows[start:]:
        if len(r) == 1:
            raw.append({"speaker": None, "text": r[0]})
        else:
            raw.append({"speaker": r[s_idx] if s_idx < len(r) else None,
                        "text": r[t_idx] if t_idx < len(r) else ""})
    return {"conversation_id": "UPLOAD", "turns": _clean_turns(raw)}


def load_txt(content: str) -> dict:
    raw = []
    for line in content.splitlines():
        if not line.strip():
            continue
        m = _LINE_RE.match(line)
        if m and m.group(1).strip().lower() in (_HANDLER_WORDS | _CUSTOMER_WORDS):
            raw.append({"speaker": m.group(1), "text": m.group(2)})
        else:
            raw.append({"speaker": None, "text": line.strip()})
    return {"conversation_id": "UPLOAD", "turns": _clean_turns(raw)}


def load_conversation(content: str, filename: str = "") -> dict:
    """Dispatch on file extension (falls back to sniffing)."""
    name = (filename or "").lower()
    if name.endswith(".json"):
        return load_json(content)
    if name.endswith(".csv"):
        return load_csv(content)
    if name.endswith(".txt"):
        return load_txt(content)
    # Sniff: JSON if it parses, else line-based text.
    stripped = content.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            return load_json(content)
        except json.JSONDecodeError:
            pass
    return load_txt(content)
