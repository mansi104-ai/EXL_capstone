"""SQLAlchemy models for GuardianCX persistence.

Works against PostgreSQL (via GUARDIANCX_DATABASE_URL) or a local SQLite file
(default). Two record types:

* EvidenceRecord — one immutable, hash-chained entry per detection/decision
  (detection, adaptation offered, citations, model confidence, guardrails,
  user decision). Append-only; each row embeds the previous row's hash.
* ConversationRecord — a whole conversation, saved from the live console or
  seeded synthetically, so a call can be replayed, reviewed and compared long
  after the session that produced it has gone.
* AuditEvent — a flat audit log of everything the system and its users did.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class EvidenceRecord(Base):
    __tablename__ = "evidence_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    record_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)

    conversation_id: Mapped[str] = mapped_column(String(64), index=True)
    customer_id: Mapped[str] = mapped_column(String(64), index=True, default="")
    turn_index: Mapped[int] = mapped_column(Integer, default=0)

    triggered_drivers: Mapped[list] = mapped_column(JSON, default=list)
    max_score: Mapped[float] = mapped_column(Float, default=0.0)
    risk_level: Mapped[str] = mapped_column(String(16), default="low")

    # Finance-domain classification for this turn — what makes the record
    # auditable against a specific obligation rather than a general duty of care.
    journey: Mapped[str] = mapped_column(String(32), default="", index=True)
    product: Mapped[str] = mapped_column(String(32), default="")
    stress_indicators: Mapped[list] = mapped_column(JSON, default=list)

    # How the customer sounded, and on which channel.
    channel: Mapped[str] = mapped_column(String(16), default="chat")
    sentiment_emotion: Mapped[str] = mapped_column(String(24), default="")
    sentiment_distress: Mapped[float] = mapped_column(Float, default=0.0)

    recommendation: Mapped[str] = mapped_column(Text, default="")
    citations: Mapped[list] = mapped_column(JSON, default=list)
    model_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    detection_source: Mapped[str] = mapped_column(String(32), default="heuristic")

    guardrails: Mapped[list] = mapped_column(JSON, default=list)
    approval_status: Mapped[str] = mapped_column(String(24), default="not_required")
    decided_by: Mapped[str] = mapped_column(String(64), default="")
    decision_note: Mapped[str] = mapped_column(Text, default="")

    masked_text: Mapped[str] = mapped_column(Text, default="")

    prev_hash: Mapped[str] = mapped_column(String(64), default="0" * 64)
    record_hash: Mapped[str] = mapped_column(String(64), default="")


class ConversationRecord(Base):
    """A saved conversation — the live console's sessions, and the seed set.

    Storing the turns rather than only the per-turn evidence is what lets a
    reviewer replay a call as it happened: the customer's words, the handler's
    replies, and the channel each turn arrived on. Only masked text is stored.
    """

    __tablename__ = "conversation_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    customer_id: Mapped[str] = mapped_column(String(64), index=True, default="")
    customer_name: Mapped[str] = mapped_column(String(120), default="")
    product: Mapped[str] = mapped_column(String(48), default="")
    channel: Mapped[str] = mapped_column(String(16), default="chat")
    origin: Mapped[str] = mapped_column(String(16), default="live", index=True)  # live | synthetic

    turns: Mapped[list] = mapped_column(JSON, default=list)

    # Denormalised summary so the library lists without replaying every turn.
    turn_count: Mapped[int] = mapped_column(Integer, default=0)
    max_risk: Mapped[str] = mapped_column(String(16), default="low")
    drivers: Mapped[list] = mapped_column(JSON, default=list)
    journeys: Mapped[list] = mapped_column(JSON, default=list)
    peak_distress: Mapped[float] = mapped_column(Float, default=0.0)
    note: Mapped[str] = mapped_column(Text, default="")


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    actor: Mapped[str] = mapped_column(String(64), default="system")
    action: Mapped[str] = mapped_column(String(64), index=True)
    conversation_id: Mapped[str] = mapped_column(String(64), default="", index=True)
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    immutable: Mapped[bool] = mapped_column(Boolean, default=True)
