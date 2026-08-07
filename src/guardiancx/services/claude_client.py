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


def _extract_json(content: Optional[str]) -> Optional[dict[str, Any]]:
    """Best-effort parse of a JSON object from model output (handles prose /
    markdown fences by extracting the outermost {...})."""
    if not content:
        return None
    try:
        return json.loads(content)
    except Exception:  # noqa: BLE001
        pass
    start, end = content.find("{"), content.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(content[start:end + 1])
        except Exception:  # noqa: BLE001
            return None
    return None


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
        # OpenAI-compatible JSON. Embed the schema in the prompt so any model
        # returns a parseable object even without native json_schema support.
        # Some OpenRouter routes reject response_format, so retry without it and
        # extract the JSON object from the text.
        sys_prompt = (
            f"{system}\n\nRespond ONLY with a single JSON object that matches "
            f"this JSON schema — no prose, no markdown fences:\n{json.dumps(schema)}"
        )
        messages = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user}]
        for use_response_format in (True, False):
            try:
                kwargs: dict[str, Any] = dict(model=self.model, max_tokens=max_tokens,
                                              messages=messages)
                if use_response_format:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = self._openai.chat.completions.create(**kwargs)
                parsed = _extract_json(resp.choices[0].message.content)
                if parsed is not None:
                    return parsed
            except Exception as exc:  # noqa: BLE001
                log.warning("OpenRouter structured attempt (response_format=%s) failed: %s",
                            use_response_format, exc)
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
