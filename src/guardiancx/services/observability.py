"""Observability + experiment logging for GuardianCX.

Thin, dependency-optional wrappers over Langfuse (tracing) and MLflow /
OpenTelemetry (metrics). When neither is configured, everything is a no-op that
still records events to an in-memory ring buffer the Settings page can display —
so the "Observability" story is visible even without external backends.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Optional

from config.settings import Settings, get_settings

from ..utils.logging import get_logger

log = get_logger("services.observability")


class Observability:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.events: "deque[dict[str, Any]]" = deque(maxlen=500)
        self._langfuse = None
        self._mlflow = None
        self.backends: list[str] = ["in-memory"]

        if self.settings.langfuse_enabled:
            try:
                from langfuse import Langfuse

                self._langfuse = Langfuse(
                    public_key=self.settings.langfuse_public_key,
                    secret_key=self.settings.langfuse_secret_key,
                    host=self.settings.langfuse_host,
                )
                self.backends.append("langfuse")
            except Exception as exc:  # noqa: BLE001
                log.warning("Langfuse unavailable: %s", exc)

        if self.settings.mlflow_enabled:
            try:
                import mlflow

                mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
                self._mlflow = mlflow
                self.backends.append("mlflow")
            except Exception as exc:  # noqa: BLE001
                log.warning("MLflow unavailable: %s", exc)

    def trace(self, name: str, **data: Any) -> None:
        """Record a pipeline span/event."""
        from ..utils.types import now_utc

        record = {"ts": now_utc().isoformat(), "name": name, **data}
        self.events.append(record)
        if self._langfuse is not None:
            try:
                self._langfuse.trace(name=name, metadata=data)
            except Exception as exc:  # noqa: BLE001
                log.debug("langfuse trace failed: %s", exc)

    def metric(self, key: str, value: float) -> None:
        if self._mlflow is not None:
            try:
                self._mlflow.log_metric(key, value)
            except Exception as exc:  # noqa: BLE001
                log.debug("mlflow metric failed: %s", exc)
        self.trace("metric", key=key, value=value)


_OBS: Observability | None = None


def get_observability() -> Observability:
    global _OBS
    if _OBS is None:
        _OBS = Observability()
    return _OBS
