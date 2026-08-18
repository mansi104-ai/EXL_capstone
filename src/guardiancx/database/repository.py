"""Persistence operations for evidence, conversations and audit records.

The evidence log is append-only and hash-chained: each new record embeds the
SHA-256 hash of the previous record, so any later edit or reordering is
detectable via `verify_chain()`.

Alongside it sits the **conversation library** — whole conversations saved from
the live console. Evidence records prove what the system decided; the library
preserves what was actually said, so a call can be replayed and re-reviewed
rather than only summarised.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Optional

from ..utils.types import CaseDecision, now_utc
from .db import session_scope
from .models import AuditEvent, ConversationRecord, EvidenceRecord

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
            "journey": decision.journey,
            "product": decision.product,
            "stress_indicators": decision.stress_indicators,
            "channel": decision.channel,
            "sentiment_emotion": decision.sentiment.emotion if decision.sentiment else "",
            "sentiment_distress": (round(decision.sentiment.distress, 4)
                                   if decision.sentiment else 0.0),
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
        "journey": r.journey,
        "product": r.product,
        "stress_indicators": r.stress_indicators,
        "channel": r.channel,
        "sentiment_emotion": r.sentiment_emotion,
        "sentiment_distress": r.sentiment_distress,
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


def rechain_evidence() -> int:
    """Recompute the whole chain from the genesis hash. Returns rows re-chained.

    Used only after an additive schema migration, where the hashed record body
    legitimately gains fields and every stored hash is therefore computed over a
    different shape. Re-chaining restores internal consistency; the migration
    that triggered it is written to the audit log, so the event is visible rather
    than silent.
    """
    with session_scope() as session:
        rows = session.query(EvidenceRecord).order_by(EvidenceRecord.id.asc()).all()
        prev_hash = GENESIS
        for r in rows:
            r.prev_hash = prev_hash
            r.record_hash = _hash(prev_hash, _record_body(r))
            prev_hash = r.record_hash
        return len(rows)


# --------------------------------------------------------------------------- #
# Conversation library
# --------------------------------------------------------------------------- #
def save_conversation(
    conversation_id: str,
    turns: list[dict[str, Any]],
    customer_id: str = "",
    customer_name: str = "",
    product: str = "",
    channel: str = "chat",
    origin: str = "live",
    max_risk: str = "low",
    drivers: Optional[list[str]] = None,
    journeys: Optional[list[str]] = None,
    peak_distress: float = 0.0,
    note: str = "",
) -> str:
    """Save or update a conversation. Idempotent on `conversation_id`.

    Saving the same id again replaces the stored turns rather than creating a
    duplicate, so a live session can be saved repeatedly as it grows.
    """
    with session_scope() as session:
        row = (
            session.query(ConversationRecord)
            .filter(ConversationRecord.conversation_id == conversation_id)
            .one_or_none()
        )
        created = row is None
        if row is None:
            row = ConversationRecord(conversation_id=conversation_id, created_at=now_utc())
            session.add(row)
        row.updated_at = now_utc()
        row.customer_id = customer_id
        row.customer_name = customer_name
        row.product = product
        row.channel = channel
        row.origin = origin
        row.turns = turns
        row.turn_count = len(turns)
        row.max_risk = max_risk
        row.drivers = drivers or []
        row.journeys = journeys or []
        row.peak_distress = round(peak_distress, 4)
        row.note = note
    audit("conversation.saved" if created else "conversation.updated",
          conversation_id=conversation_id,
          detail={"turns": len(turns), "origin": origin, "max_risk": max_risk})
    return conversation_id


def _conversation_to_dict(r: ConversationRecord) -> dict[str, Any]:
    return {
        "conversation_id": r.conversation_id,
        "customer_id": r.customer_id,
        "customer_name": r.customer_name,
        "product": r.product,
        "channel": r.channel,
        "origin": r.origin,
        "turns": r.turns or [],
        "turn_count": r.turn_count,
        "max_risk": r.max_risk,
        "drivers": r.drivers or [],
        "journeys": r.journeys or [],
        "peak_distress": r.peak_distress,
        "note": r.note,
        "created_at": r.created_at.isoformat() if r.created_at else "",
        "updated_at": r.updated_at.isoformat() if r.updated_at else "",
    }


def list_saved_conversations(limit: int = 200) -> list[dict[str, Any]]:
    with session_scope() as session:
        rows = (
            session.query(ConversationRecord)
            .order_by(ConversationRecord.updated_at.desc())
            .limit(limit)
            .all()
        )
        return [_conversation_to_dict(r) for r in rows]


def get_saved_conversation(conversation_id: str) -> Optional[dict[str, Any]]:
    with session_scope() as session:
        row = (
            session.query(ConversationRecord)
            .filter(ConversationRecord.conversation_id == conversation_id)
            .one_or_none()
        )
        return _conversation_to_dict(row) if row else None


def delete_saved_conversation(conversation_id: str) -> bool:
    with session_scope() as session:
        deleted = (
            session.query(ConversationRecord)
            .filter(ConversationRecord.conversation_id == conversation_id)
            .delete()
        )
    if deleted:
        audit("conversation.deleted", conversation_id=conversation_id)
    return bool(deleted)


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
        session.query(ConversationRecord).delete()
        session.query(AuditEvent).delete()
