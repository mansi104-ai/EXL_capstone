"""Parse the firm's vulnerability policy markdown into retrievable sections.

Each section corresponds to one policy clause (e.g. VP-L1 Bereavement), tagged
with the driver it sits under, so retrieval can be scoped to a detected driver.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..schemas import Driver

_DRIVER_HEADER = re.compile(r"^##\s+Driver:\s+(.+?)\s*$", re.IGNORECASE)
_SECTION_HEADER = re.compile(r"^###\s+(\S+)\s+[—-]\s+(.+?)\s*$")

_DRIVER_NAME_MAP = {
    "health": Driver.HEALTH,
    "life events": Driver.LIFE_EVENTS,
    "resilience": Driver.RESILIENCE,
    "capability": Driver.CAPABILITY,
}


@dataclass
class PolicySection:
    code: str          # e.g. "VP-L1"
    title: str         # e.g. "Bereavement"
    driver: Driver
    body: str

    @property
    def text(self) -> str:
        """Full text used for embedding/retrieval."""
        return f"{self.code} {self.title}. {self.body}"


def parse_policy(path: str | Path) -> list[PolicySection]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    sections: list[PolicySection] = []
    current_driver: Driver | None = None
    code = title = None
    body_lines: list[str] = []

    def flush():
        nonlocal code, title, body_lines
        if code and title and current_driver is not None:
            sections.append(
                PolicySection(
                    code=code,
                    title=title,
                    driver=current_driver,
                    body=" ".join(l.strip() for l in body_lines if l.strip()),
                )
            )
        code = title = None
        body_lines = []

    for line in lines:
        dm = _DRIVER_HEADER.match(line)
        if dm:
            flush()
            name = dm.group(1).strip().lower()
            current_driver = _DRIVER_NAME_MAP.get(name)
            continue
        sm = _SECTION_HEADER.match(line)
        if sm:
            flush()
            code, title = sm.group(1).strip(), sm.group(2).strip()
            continue
        # Stop capturing body at a cross-cutting/other top-level header.
        if line.startswith("## ") and not _DRIVER_HEADER.match(line):
            flush()
            current_driver = None
            continue
        if code:
            if line.strip() == "---":  # skip markdown horizontal rules
                continue
            body_lines.append(line)

    flush()
    return sections
