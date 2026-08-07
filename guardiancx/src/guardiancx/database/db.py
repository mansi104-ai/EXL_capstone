"""Database engine/session management for GuardianCX."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

from sqlalchemy import create_engine
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
    _SESSION_FACTORY = sessionmaker(bind=_ENGINE, expire_on_commit=False, future=True)
    log.info("Database ready (%s)", "sqlite" if url.startswith("sqlite") else "postgres")
    return _ENGINE


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
