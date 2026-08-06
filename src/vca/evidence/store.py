"""Append-only, hash-chained evidence store.

Every detection + guidance offered + handler outcome is written as one immutable
JSONL record. Each record embeds the SHA-256 hash of the previous record, so the
log forms a tamper-evident chain: altering any past record breaks verification.

This is what lets the firm demonstrate consistent, fair treatment at portfolio
level and prove the evidence was not edited after the fact.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

from ..schemas import Detection, EvidenceRecord, Guidance, Outcome

GENESIS_HASH = "0" * 64


def _canonical(payload: dict) -> str:
    """Deterministic JSON for hashing (sorted keys, no whitespace)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _hash_record(prev_hash: str, body: dict) -> str:
    return hashlib.sha256((prev_hash + _canonical(body)).encode("utf-8")).hexdigest()


class EvidenceStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # --- writing ---------------------------------------------------------
    def _last_hash(self) -> str:
        last = GENESIS_HASH
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        last = json.loads(line)["record_hash"]
        return last

    def append(self, detection: Detection, guidance: Guidance, outcome: Outcome) -> EvidenceRecord:
        prev_hash = self._last_hash()
        record_id = str(uuid.uuid4())
        body = {
            "record_id": record_id,
            "conversation_id": detection.conversation_id,
            "turn_index": detection.turn_index,
            "detection": detection.model_dump(mode="json"),
            "guidance": guidance.model_dump(mode="json"),
            "outcome": outcome.model_dump(mode="json"),
        }
        record_hash = _hash_record(prev_hash, body)
        record = EvidenceRecord(
            record_id=record_id,
            conversation_id=detection.conversation_id,
            turn_index=detection.turn_index,
            detection=detection,
            guidance=guidance,
            outcome=outcome,
            prev_hash=prev_hash,
            record_hash=record_hash,
        )
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(record.model_dump_json() + "\n")
        return record

    # --- reading / verification -----------------------------------------
    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        records = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def verify_chain(self) -> tuple[bool, str | None]:
        """Return (ok, first_bad_record_id). Detects tampering/reordering."""
        prev_hash = GENESIS_HASH
        for rec in self.read_all():
            body = {
                "record_id": rec["record_id"],
                "conversation_id": rec["conversation_id"],
                "turn_index": rec["turn_index"],
                "detection": rec["detection"],
                "guidance": rec["guidance"],
                "outcome": rec["outcome"],
            }
            if rec["prev_hash"] != prev_hash:
                return False, rec["record_id"]
            if _hash_record(prev_hash, body) != rec["record_hash"]:
                return False, rec["record_id"]
            prev_hash = rec["record_hash"]
        return True, None
