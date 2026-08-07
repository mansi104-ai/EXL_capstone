"""LLM client for GuardianCX.

Provides the agents' reasoning via a pluggable backend, resolved in this order:

  1. **Anthropic / Claude** — if ANTHROPIC_API_KEY is set (uses `claude-opus-4-8`
     with structured JSON via `output_config.format`).
  2. **OpenRouter** — if OPENROUTER_API_KEY is set (OpenAI-compatible; JSON via
     `response_format`). Lets you run any OpenRouter model, including Claude,
     without a first-party Anthropic key.
  3. **Heuristic fallback** — no key: `available` is False and the agents use
     their deterministic fallbacks, so the app still runs.

The public surface (`available`, `model`, `provider`, `structured()`, `text()`)
is identical regardless of backend, so the agents don't care which is active.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from config.settings import Settings, get_settings

from ..utils.logging import get_logger

log = get_logger("services.llm")


class LLMClient:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.provider: str = "heuristic"
        self._anthropic = None
        self._openai = None

        if self.settings.claude_enabled:
            try:
                import anthropic

                self._anthropic = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
                self.provider = "anthropic"
            except Exception as exc:  # noqa: BLE001
                log.warning("Anthropic init failed: %s", exc)

        if self._anthropic is None and self.settings.openrouter_enabled:
            try:
                from openai import OpenAI

                self._openai = OpenAI(
                    api_key=self.settings.openrouter_api_key,
                    base_url=self.settings.openrouter_base_url,
                )
                self.provider = "openrouter"
            except Exception as exc:  # noqa: BLE001
                log.warning("OpenRouter init failed: %s", exc)

    # --- capability ------------------------------------------------------
    @property
    def available(self) -> bool:
        return self._anthropic is not None or self._openai is not None

    @property
    def model(self) -> str:
        if self.provider == "openrouter":
            return self.settings.openrouter_model
        return self.settings.guardiancx_claude_model

    @property
    def provider_label(self) -> str:
        return {
            "anthropic": f"Claude ({self.model})",
            "openrouter": f"OpenRouter ({self.model})",
            "heuristic": "heuristic (no LLM key)",
        }[self.provider]

    # --- structured JSON -------------------------------------------------
    def structured(self, system: str, user: str, schema: dict[str, Any],
                   max_tokens: int = 1500) -> Optional[dict[str, Any]]:
        if self._anthropic is not None:
            return self._anthropic_structured(system, user, schema, max_tokens)
        if self._openai is not None:
            return self._openrouter_structured(system, user, schema, max_tokens)
        return None

    def _anthropic_structured(self, system, user, schema, max_tokens):
        try:
            resp = self._anthropic.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"format": {"type": "json_schema", "schema": schema}},
            )
            text = next((b.text for b in resp.content if b.type == "text"), "")
            return json.loads(text)
        except Exception as exc:  # noqa: BLE001
            log.warning("Anthropic structured call failed, falling back: %s", exc)
            return None

    def _openrouter_structured(self, system, user, schema, max_tokens):
        # OpenAI-compatible JSON mode. Embed the schema in the prompt so any
        # model returns a parseable object even without native json_schema support.
        sys_prompt = (
            f"{system}\n\nRespond ONLY with a single JSON object that matches "
            f"this JSON schema (no prose, no markdown):\n{json.dumps(schema)}"
        )
        try:
            resp = self._openai.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "system", "content": sys_prompt},
                          {"role": "user", "content": user}],
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as exc:  # noqa: BLE001
            log.warning("OpenRouter structured call failed, falling back: %s", exc)
            return None

    # --- plain text ------------------------------------------------------
    def text(self, system: str, user: str, max_tokens: int = 600) -> Optional[str]:
        try:
            if self._anthropic is not None:
                resp = self._anthropic.messages.create(
                    model=self.model, max_tokens=max_tokens, system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return next((b.text for b in resp.content if b.type == "text"), "")
            if self._openai is not None:
                resp = self._openai.chat.completions.create(
                    model=self.model, max_tokens=max_tokens,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                )
                return resp.choices[0].message.content
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM text call failed: %s", exc)
        return None


_CLIENT: LLMClient | None = None


def get_llm() -> LLMClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = LLMClient()
    return _CLIENT


# Backwards-compatible alias (agents import get_claude()).
def get_claude() -> LLMClient:
    return get_llm()
