"""Parse policy markdown into retrievable, driver-tagged chunks."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from ..utils.types import Driver

_DRIVER_HEADER = re.compile(r"^##\s+Driver:\s+(.+?)\s*$", re.IGNORECASE)
_SECTION_HEADER = re.compile(r"^###\s+(\S+)\s+[—-]\s+(.+?)\s*$")

_DRIVER_MAP = {
    "health": Driver.HEALTH,
    "life events": Driver.LIFE_EVENTS,
    "resilience": Driver.RESILIENCE,
    "capability": Driver.CAPABILITY,
}


class Chunk:
    def __init__(self, ref: str, title: str, driver: Optional[Driver], text: str, source: str):
        self.ref = ref
        self.title = title
        self.driver = driver
        self.text = text
        self.source = source

    @property
    def embed_text(self) -> str:
        return f"{self.ref} {self.title}. {self.text}"


def chunk_policy_file(path: str | Path) -> list[Chunk]:
    p = Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    chunks: list[Chunk] = []
    driver: Optional[Driver] = None
    ref = title = None
    body: list[str] = []

    def flush():
        nonlocal ref, title, body
        if ref and title:
            text = " ".join(l.strip() for l in body if l.strip() and l.strip() != "---")
            chunks.append(Chunk(ref, title, driver, text, p.name))
        ref = title = None
        body = []

    for line in lines:
        dm = _DRIVER_HEADER.match(line)
        if dm:
            flush()
            driver = _DRIVER_MAP.get(dm.group(1).strip().lower())
            continue
        sm = _SECTION_HEADER.match(line)
        if sm:
            flush()
            ref, title = sm.group(1).strip(), sm.group(2).strip()
            continue
        if line.startswith("## ") and not _DRIVER_HEADER.match(line):
            flush()
            driver = None
            continue
        if ref:
            body.append(line)
    flush()
    return chunks


def chunk_policy_dir(directory: str | Path) -> list[Chunk]:
    out: list[Chunk] = []
    for md in sorted(Path(directory).glob("*.md")):
        out.extend(chunk_policy_file(md))
    return out
