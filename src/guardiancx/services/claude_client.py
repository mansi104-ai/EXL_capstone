"""Claude LLM client for GuardianCX.

Wraps the Anthropic SDK (`claude-opus-4-8`) and always requests **structured
JSON** via `output_config.format` so downstream agents get validated objects,
never free text to parse. If no API key is configured (or the SDK/network is
unavailable), `available` is False and callers fall back to deterministic
heuristics — the app keeps working without Claude.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from config.settings import Settings, get_settings

from ..utils.logging import get_logger

log = get_logger("services.claude")


class ClaudeClient:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._client = None
        self._init_error: str | None = None
        if self.settings.claude_enabled:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
            except Exception as exc:  # noqa: BLE001 - degrade gracefully
                self._init_error = str(exc)
                log.warning("Claude unavailable, using fallback: %s", exc)

    @property
    def available(self) -> bool:
        return self._client is not None

    @property
    def model(self) -> str:
        return self.settings.guardiancx_claude_model

    def structured(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        max_tokens: int = 1500,
    ) -> Optional[dict[str, Any]]:
        """Return a JSON object matching `schema`, or None if Claude is unavailable.

        Uses `output_config.format` (json_schema) — the current, prefill-free way
        to constrain Claude's output. The first returned text block is guaranteed
        to be valid JSON for the given schema.
        """
        if not self.available:
            return None
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": schema,
                    }
                },
            )
            text = next((b.text for b in resp.content if b.type == "text"), "")
            return json.loads(text)
        except Exception as exc:  # noqa: BLE001 - never crash the pipeline on LLM errors
            log.warning("Claude structured call failed, falling back: %s", exc)
            return None

    def text(self, system: str, user: str, max_tokens: int = 1000) -> Optional[str]:
        """Plain-text completion (used for free-form guidance narration)."""
        if not self.available:
            return None
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return next((b.text for b in resp.content if b.type == "text"), "")
        except Exception as exc:  # noqa: BLE001
            log.warning("Claude text call failed: %s", exc)
            return None


_CLIENT: ClaudeClient | None = None


def get_claude() -> ClaudeClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = ClaudeClient()
    return _CLIENT
