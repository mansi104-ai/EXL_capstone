"""Prompt-injection detection guardrail.

Flags customer text that tries to override the agent's instructions or extract
the system prompt. Signal-based; on a hit the pipeline treats the utterance as
untrusted input (it is never followed as an instruction — the agent only ever
classifies vulnerability and cites policy).
"""
from __future__ import annotations

import re

from ..utils.types import GuardrailResult
from .base import Guardrail, GuardrailContext

_PATTERNS = [
    r"ignore\s+(?:\w+\s+){0,3}(instructions|rules|prompt)",
    r"disregard\s+(?:\w+\s+){0,3}(instructions|rules|prompt)",
    r"forget\s+(everything|the above|your instructions|all previous)",
    r"system prompt",
    r"you are now",
    r"act as (an?|the) ",
    r"reveal your (prompt|instructions|rules)",
    r"override",
    r"jailbreak",
]
_REGEX = [re.compile(p, re.IGNORECASE) for p in _PATTERNS]


class InjectionGuardrail(Guardrail):
    name = "prompt_injection"
    severity = "block"

    def check(self, ctx: GuardrailContext) -> GuardrailResult:
        hits = [p.pattern for p in _REGEX if p.search(ctx.text)]
        if not hits:
            return self._ok("No injection patterns detected.")
        return self._fail(
            f"Possible prompt injection ({len(hits)} pattern(s)). "
            "Treating utterance as untrusted; not executed as an instruction.",
            severity="block",
            patterns=hits,
        )
