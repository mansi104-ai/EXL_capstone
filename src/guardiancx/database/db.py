"""Database engine/session management for GuardianCX.

`create_all` creates missing *tables* but never adds a column to a table that
already exists, so an existing local database silently loses any field added
after it was first created. `_add_missing_columns` closes that gap with a
minimal additive migration: it reads the live schema, compares it to the models,
and issues `ALTER TABLE ... ADD COLUMN` for anything absent. Additive only — it
never drops or retypes a column, so it cannot lose data.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from config.settings import Settings, get_settings

from ..utils.logging import get_logger
from .models import Base

log = get_logger("database")

_ENGINE = None
_SESSION_FACTORY = None


def init_engine(settings: Optional[Settings] = None):
    global _ENGINE, _SESSION_FACTORY
    if _ENGINE is not None:
        return _ENGINE
    settings = settings or get_settings()
    url = settings.database_url
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    _ENGINE = create_engine(url, connect_args=connect_args, future=True)
    Base.metadata.create_all(_ENGINE)
    added = _add_missing_columns(_ENGINE)
    _SESSION_FACTORY = sessionmaker(bind=_ENGINE, expire_on_commit=False, future=True)
    log.info("Database ready (%s)", "sqlite" if url.startswith("sqlite") else "postgres")

    if added:
        # New columns join the hashed record body, so every existing hash is now
        # computed over a different set of fields. Re-chain once so the evidence
        # log stays internally consistent and tamper-evident from here on, and
        # write an audit event recording that it happened.
        from .repository import audit, rechain_evidence

        count = rechain_evidence()
        audit("schema.migrated", detail={"columns_added": added, "records_rechained": count})
        log.info("Migrated %d column(s); re-chained %d evidence record(s).", len(added), count)
    return _ENGINE


def _add_missing_columns(engine) -> list[str]:
    """Add any model column absent from the live table. Returns what was added."""
    inspector = inspect(engine)
    added: list[str] = []
    for table in Base.metadata.sorted_tables:
        if not inspector.has_table(table.name):
            continue
        existing = {c["name"] for c in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in existing:
                continue
            ddl = column.type.compile(engine.dialect)
            with engine.begin() as conn:
                conn.execute(text(
                    f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {ddl}'
                ))
            added.append(f"{table.name}.{column.name}")
    return added


@contextmanager
def session_scope() -> Iterator[Session]:
    if _SESSION_FACTORY is None:
        init_engine()
    session = _SESSION_FACTORY()  # type: ignore[misc]
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def backend_name() -> str:
    settings = get_settings()
    return "postgresql" if settings.guardiancx_database_url.startswith("postgre") else "sqlite"
