"""LLM client for GuardianCX.

Provides the agents' reasoning via a pluggable backend, resolved in this order:

  1. **Anthropic / Claude** — if ANTHROPIC_API_KEY is set (uses `claude-opus-5`
     with structured JSON via `output_config.format`).
  2. **OpenRouter** — if OPENROUTER_API_KEY is set (OpenAI-compatible; JSON via
     `response_format`). Lets you run any OpenRouter model, including Claude,
     without a first-party Anthropic key.
  3. **Heuristic fallback** — no key: `available` is False and the agents use
     their deterministic fallbacks, so the app still runs.

The public surface (`available`, `model`, `provider`, `structured()`, `text()`,
`stream_text()`) is identical regardless of backend, so the agents don't care
which is active.

**Latency control.** A live call cannot wait on deep reasoning, so every call
site declares an `effort` level: `low` for the customer-facing reply that is
being spoken back, `medium`/`high` for the classification and policy-grounding
calls behind it. Adaptive thinking is requested alongside it; if a model or
route rejects either parameter the call is retried without them, so an older
model or an OpenRouter passthrough still works.
"""
from __future__ import annotations

import json
from typing import Any, Iterator, Optional

from config.settings import Settings, get_settings

from ..utils.logging import get_logger

log = get_logger("services.llm")

# Effort levels the live pipeline uses, by call site.
EFFORT_REPLY = "low"        # spoken back to the customer — latency dominates
EFFORT_ANALYSIS = "medium"  # detection / sentiment — accuracy matters, still per-turn
EFFORT_GUIDANCE = "high"    # policy-grounded advice a handler will act on


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
        # Set False once a request is rejected for the thinking/effort parameters,
        # so we stop paying a failed round-trip on every subsequent call.
        self._tuning_supported = True

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

    # --- Anthropic request tuning ---------------------------------------
    def _tuning(self, effort: Optional[str], schema: Optional[dict] = None) -> dict[str, Any]:
        """Adaptive thinking + effort, plus a JSON schema when one is supplied."""
        kwargs: dict[str, Any] = {}
        output_config: dict[str, Any] = {}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        if self._tuning_supported:
            if effort:
                output_config["effort"] = effort
            kwargs["thinking"] = {"type": "adaptive"}
        if output_config:
            kwargs["output_config"] = output_config
        return kwargs

    @staticmethod
    def _is_tuning_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(k in msg for k in ("thinking", "effort", "output_config"))

    # --- structured JSON -------------------------------------------------
    def structured(self, system: str, user: str, schema: dict[str, Any],
                   max_tokens: int = 1500,
                   effort: str = EFFORT_ANALYSIS) -> Optional[dict[str, Any]]:
        if self._anthropic is not None:
            return self._anthropic_structured(system, user, schema, max_tokens, effort)
        if self._openai is not None:
            return self._openrouter_structured(system, user, schema, max_tokens)
        return None

    def _anthropic_structured(self, system, user, schema, max_tokens, effort):
        for attempt in range(2):
            try:
                resp = self._anthropic.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    **self._tuning(effort, schema),
                )
                text = next((b.text for b in resp.content if b.type == "text"), "")
                return _extract_json(text)
            except Exception as exc:  # noqa: BLE001
                if attempt == 0 and self._tuning_supported and self._is_tuning_error(exc):
                    log.info("Model rejected thinking/effort; retrying without them (%s).", exc)
                    self._tuning_supported = False
                    continue
                log.warning("Anthropic structured call failed, falling back: %s", exc)
                return None
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
    def text(self, system: str, user: str, max_tokens: int = 600,
             effort: str = EFFORT_REPLY) -> Optional[str]:
        """Non-streaming completion. Prefer `stream_text` for anything a person
        is waiting on."""
        chunks = list(self.stream_text(system, user, max_tokens=max_tokens, effort=effort))
        return "".join(chunks) or None

    # --- streaming text --------------------------------------------------
    def stream_text(self, system: str, user: str, max_tokens: int = 600,
                    effort: str = EFFORT_REPLY) -> Iterator[str]:
        """Yield the reply incrementally as the model produces it.

        Streaming is what makes the live console feel like a conversation rather
        than a form submission: the handler sees the draft forming, and the
        speech synthesiser can start on the first sentence instead of waiting for
        the last one. Yields nothing when no LLM is configured — callers fall
        back to their deterministic template.
        """
        if self._anthropic is not None:
            yield from self._anthropic_stream(system, user, max_tokens, effort)
        elif self._openai is not None:
            yield from self._openrouter_stream(system, user, max_tokens)

    def _anthropic_stream(self, system, user, max_tokens, effort) -> Iterator[str]:
        for attempt in range(2):
            produced = False
            try:
                with self._anthropic.messages.stream(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    **self._tuning(effort),
                ) as stream:
                    for piece in stream.text_stream:
                        produced = True
                        yield piece
                return
            except Exception as exc:  # noqa: BLE001
                # Only safe to retry when nothing was emitted yet — otherwise the
                # caller would see the opening of the reply twice.
                if (attempt == 0 and not produced and self._tuning_supported
                        and self._is_tuning_error(exc)):
                    log.info("Model rejected thinking/effort; retrying without them (%s).", exc)
                    self._tuning_supported = False
                    continue
                log.warning("Anthropic stream failed: %s", exc)
                return

    def _openrouter_stream(self, system, user, max_tokens) -> Iterator[str]:
        try:
            stream = self._openai.chat.completions.create(
                model=self.model, max_tokens=max_tokens,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                stream=True,
            )
            for event in stream:
                if not event.choices:
                    continue
                piece = event.choices[0].delta.content
                if piece:
                    yield piece
        except Exception as exc:  # noqa: BLE001
            log.warning("OpenRouter stream failed: %s", exc)


_CLIENT: LLMClient | None = None


def get_llm() -> LLMClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = LLMClient()
    return _CLIENT


# Backwards-compatible alias (agents import get_claude()).
def get_claude() -> LLMClient:
    return get_llm()
