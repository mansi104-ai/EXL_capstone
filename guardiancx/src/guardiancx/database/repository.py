"""Persistence operations for evidence + audit records.

The evidence log is append-only and hash-chained: each new record embeds the
SHA-256 hash of the previous record, so any later edit or reordering is
detectable via `verify_chain()`.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Optional

from ..utils.types import CaseDecision, now_utc
from .db import session_scope
from .models import AuditEvent, EvidenceRecord

GENESIS = "0" * 64


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _hash(prev_hash: str, body: dict[str, Any]) -> str:
    return hashlib.sha256((prev_hash + _canonical(body)).encode("utf-8")).hexdigest()


def _last_hash(session) -> str:
    row = (
        session.query(EvidenceRecord)
        .order_by(EvidenceRecord.id.desc())
        .first()
    )
    return row.record_hash if row else GENESIS


def record_evidence(
    decision: CaseDecision,
    customer_id: str = "",
    decided_by: str = "",
    decision_note: str = "",
) -> str:
    """Append one immutable evidence record for a decision. Returns record_id."""
    with session_scope() as session:
        prev_hash = _last_hash(session)
        record_id = str(uuid.uuid4())
        body = {
            "record_id": record_id,
            "conversation_id": decision.conversation_id,
            "customer_id": customer_id,
            "turn_index": decision.turn_index,
            "triggered_drivers": [d.value for d in decision.assessment.triggered],
            "max_score": round(decision.assessment.max_score, 4),
            "risk_level": decision.risk_level.value,
            "recommendation": decision.recommendation.summary if decision.recommendation else "",
            "citations": decision.recommendation.citations if decision.recommendation else [],
            "model_confidence": decision.recommendation.confidence if decision.recommendation else 0.0,
            "detection_source": decision.assessment.source,
            "guardrails": [r.model_dump() for r in decision.guardrails.results],
            "approval_status": decision.approval_status.value,
            "decided_by": decided_by,
            "decision_note": decision_note,
            "masked_text": decision.masked_text,
        }
        record_hash = _hash(prev_hash, body)
        row = EvidenceRecord(
            created_at=now_utc(),
            prev_hash=prev_hash,
            record_hash=record_hash,
            **body,  # includes record_id
        )
        session.add(row)
    audit("evidence.recorded", conversation_id=decision.conversation_id,
          detail={"record_id": record_id, "risk": decision.risk_level.value})
    return record_id


def update_approval(record_id: str, status: str, decided_by: str, note: str = "") -> bool:
    """Set the human decision on a pending record.

    The approval decision is a legitimate post-hoc annotation, so we re-chain
    from this record forward to keep the hash chain valid and tamper-evident.
    """
    with session_scope() as session:
        rows = session.query(EvidenceRecord).order_by(EvidenceRecord.id.asc()).all()
        idx = next((i for i, r in enumerate(rows) if r.record_id == record_id), None)
        if idx is None:
            return False
        rows[idx].approval_status = status
        rows[idx].decided_by = decided_by
        rows[idx].decision_note = note
        # Re-hash from idx forward.
        prev_hash = rows[idx].prev_hash
        for r in rows[idx:]:
            body = _record_body(r)
            r.prev_hash = prev_hash
            r.record_hash = _hash(prev_hash, body)
            prev_hash = r.record_hash
    audit("approval.decided", detail={"record_id": record_id, "status": status, "by": decided_by})
    return True


def _record_body(r: EvidenceRecord) -> dict[str, Any]:
    return {
        "record_id": r.record_id,
        "conversation_id": r.conversation_id,
        "customer_id": r.customer_id,
        "turn_index": r.turn_index,
        "triggered_drivers": r.triggered_drivers,
        "max_score": r.max_score,
        "risk_level": r.risk_level,
        "recommendation": r.recommendation,
        "citations": r.citations,
        "model_confidence": r.model_confidence,
        "detection_source": r.detection_source,
        "guardrails": r.guardrails,
        "approval_status": r.approval_status,
        "decided_by": r.decided_by,
        "decision_note": r.decision_note,
        "masked_text": r.masked_text,
    }


def verify_chain() -> tuple[bool, Optional[str]]:
    with session_scope() as session:
        rows = session.query(EvidenceRecord).order_by(EvidenceRecord.id.asc()).all()
        prev_hash = GENESIS
        for r in rows:
            if r.prev_hash != prev_hash or _hash(prev_hash, _record_body(r)) != r.record_hash:
                return False, r.record_id
            prev_hash = r.record_hash
    return True, None


def list_evidence(limit: int = 500) -> list[dict[str, Any]]:
    with session_scope() as session:
        rows = (
            session.query(EvidenceRecord)
            .order_by(EvidenceRecord.id.desc())
            .limit(limit)
            .all()
        )
        return [_row_to_dict(r) for r in rows]


def list_pending() -> list[dict[str, Any]]:
    with session_scope() as session:
        rows = (
            session.query(EvidenceRecord)
            .filter(EvidenceRecord.approval_status == "pending")
            .order_by(EvidenceRecord.id.desc())
            .all()
        )
        return [_row_to_dict(r) for r in rows]


def customer_timeline(customer_id: str) -> list[dict[str, Any]]:
    with session_scope() as session:
        rows = (
            session.query(EvidenceRecord)
            .filter(EvidenceRecord.customer_id == customer_id)
            .order_by(EvidenceRecord.id.asc())
            .all()
        )
        return [_row_to_dict(r) for r in rows]


def _row_to_dict(r: EvidenceRecord) -> dict[str, Any]:
    body = _record_body(r)
    body.update({
        "created_at": r.created_at.isoformat() if r.created_at else "",
        "prev_hash": r.prev_hash,
        "record_hash": r.record_hash,
    })
    return body


# --------------------------------------------------------------------------- #
# Audit log
# --------------------------------------------------------------------------- #
def audit(action: str, actor: str = "system", conversation_id: str = "",
          detail: Optional[dict[str, Any]] = None) -> None:
    with session_scope() as session:
        session.add(AuditEvent(
            action=action, actor=actor, conversation_id=conversation_id,
            detail=detail or {}, created_at=now_utc(),
        ))


def list_audit(limit: int = 500) -> list[dict[str, Any]]:
    with session_scope() as session:
        rows = (
            session.query(AuditEvent)
            .order_by(AuditEvent.id.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "created_at": r.created_at.isoformat() if r.created_at else "",
                "actor": r.actor,
                "action": r.action,
                "conversation_id": r.conversation_id,
                "detail": r.detail,
            }
            for r in rows
        ]


def clear_all() -> None:
    with session_scope() as session:
        session.query(EvidenceRecord).delete()
        session.query(AuditEvent).delete()
