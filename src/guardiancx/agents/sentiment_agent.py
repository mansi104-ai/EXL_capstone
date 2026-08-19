"""Sentiment Agent.

Sentiment here is not a product review score. It exists because FG21/1 expects a
firm to notice when a customer is struggling *in the moment* and adapt the call —
slow down, stop the collections script, offer a callback — and because the
handler needs that read on screen while the call is still happening, not in a
QA report next week.

Two inputs, deliberately:

* **What was said** — the masked utterance.
* **How it was said** — the acoustic measurements from `voice.prosody`, when the
  turn came in over the call channel.

The fusion is asymmetric on purpose. Words can establish distress on their own;
voice can only corroborate. A raised voice is not a vulnerability, and a system
that inferred one from loudness would systematically mis-serve people who are
simply angry, hard of hearing, or on a bad line. So the acoustic reading can
raise a text-derived distress score and can flag an escalation for a human to
look at — it can never create a finding by itself.

Primary path: the LLM with a strict schema. Fallback: a lexicon-and-signal blend,
so a reading always exists.
"""
from __future__ import annotations

import re

from ..services.claude_client import EFFORT_ANALYSIS, get_claude
from ..utils.types import SentimentReading
from ..voice.prosody import VoiceSignals
from .prompts import SENTIMENT_SYSTEM
from .state import AgentState, recent_history

_SCHEMA = {
    "type": "object",
    "properties": {
        "valence": {"type": "number"},
        "arousal": {"type": "number"},
        "distress": {"type": "number"},
        "emotion": {
            "type": "string",
            "enum": ["calm", "anxious", "frustrated", "angry", "sad", "distressed",
                     "confused", "relieved", "hopeful"],
        },
        "escalate": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
    "required": ["valence", "arousal", "distress", "emotion", "escalate", "rationale"],
    "additionalProperties": False,
}

# Lexicon for the no-LLM path. Weighted by how strongly each phrase indicates
# distress rather than mere negativity.
_DISTRESS_LEXICON: list[tuple[float, str, list[str]]] = [
    (0.95, "distressed", ["can't go on", "cant go on", "can't carry on", "cant carry on",
                          "how i'll carry on", "how ill carry on", "end it all", "no way out",
                          "want to die", "harm myself", "hurt myself", "no point in",
                          "give up completely", "hopeless"]),
    (0.80, "distressed", ["i'm desperate", "im desperate", "at breaking point",
                          "don't know what i'd do", "dont know what id do",
                          "don't know what to do", "terrified", "petrified",
                          "crying", "in tears", "can't cope", "cant cope",
                          "not coping"]),
    (0.65, "sad", ["passed away", "died", "lost my", "heartbroken", "devastated",
                   "so alone", "on my own now"]),
    (0.60, "anxious", ["worried sick", "so worried", "panicking", "scared",
                       "afraid", "anxious", "keeps me awake", "dreading"]),
    (0.55, "frustrated", ["fed up", "sick of", "no one listens", "keep being passed",
                          "third time i've called", "getting nowhere"]),
    (0.55, "confused", ["don't understand", "dont understand", "confusing",
                        "plain english", "a bit lost", "makes no sense"]),
    (0.70, "angry", ["useless", "disgusting", "outrageous", "furious", "appalling"]),
    (0.00, "relieved", ["thank you so much", "that's a relief", "thats a relief",
                        "that helps", "much better"]),
]

_POSITIVE = ["thank you", "thanks", "great", "perfect", "appreciate", "helpful",
             "relief", "sorted", "that's great"]
_NEGATIVE = ["can't", "cannot", "struggling", "behind", "worried", "sorry",
             "problem", "difficult", "hard", "no", "not", "won't", "afraid"]


def _heuristic(text: str, voice: VoiceSignals) -> SentimentReading:
    low = text.lower()

    distress, emotion = 0.0, "calm"
    for weight, label, phrases in _DISTRESS_LEXICON:
        if any(p in low for p in phrases):
            if weight > distress:
                distress, emotion = weight, label
            elif weight == 0.0 and distress == 0.0:
                emotion = label

    pos = sum(1 for p in _POSITIVE if p in low)
    neg = sum(1 for n in _NEGATIVE if re.search(r"\b" + re.escape(n) + r"\b", low))
    total = pos + neg
    valence = ((pos - neg) / total) if total else 0.0

    arousal = min(1.0, 0.2 + 0.15 * neg + (0.5 if "!" in text else 0.0))

    if voice.available:
        # Corroboration only: acoustic distress can lift a text score by up to
        # 0.25, and can set a floor of 0.35 when the words are neutral but the
        # voice plainly is not — enough to surface it to a human, never enough to
        # constitute a finding on its own.
        distress = min(1.0, distress + 0.25 * voice.distress)
        if voice.distress >= 0.6:
            distress = max(distress, 0.35)
        arousal = min(1.0, max(arousal, voice.agitation))
        if distress >= 0.5 and emotion == "calm":
            emotion = "anxious"

    return SentimentReading(
        valence=round(max(-1.0, min(1.0, valence)), 3),
        arousal=round(arousal, 3),
        distress=round(distress, 3),
        emotion=emotion,
        escalate=distress >= 0.75,
        rationale="Lexicon and acoustic-signal blend (no LLM configured).",
        source="heuristic",
        voice=voice.model_dump() if voice.available else None,
    )


def run(state: AgentState) -> AgentState:
    trace = state.setdefault("trace", [])
    if state.get("speaker") != "customer":
        state["sentiment"] = SentimentReading(source="skipped")
        trace.append({"agent": "sentiment", "summary": "Agent turn — not assessed."})
        return state

    text = state.get("masked_text") or state.get("text", "")
    voice = VoiceSignals(**state["voice_signals"]) if state.get("voice_signals") else VoiceSignals()

    llm = get_claude()
    history = recent_history(state, turns=4)
    user = ((history + "\n\n") if history else "") + \
        f"Customer utterance: {text}\nVoice signals: {voice.summary()}"
    result = llm.structured(SENTIMENT_SYSTEM, user, _SCHEMA,
                            max_tokens=400, effort=EFFORT_ANALYSIS)
    if result:
        reading = SentimentReading(
            valence=round(max(-1.0, min(1.0, float(result.get("valence", 0.0)))), 3),
            arousal=round(max(0.0, min(1.0, float(result.get("arousal", 0.0)))), 3),
            distress=round(max(0.0, min(1.0, float(result.get("distress", 0.0)))), 3),
            emotion=str(result.get("emotion", "calm")),
            escalate=bool(result.get("escalate")),
            rationale=str(result.get("rationale", "")),
            source=llm.provider,
            voice=voice.model_dump() if voice.available else None,
        )
        # The acoustic floor applies whichever path produced the reading: if the
        # voice is plainly distressed, that reaches a human even when the words
        # were measured.
        if voice.available and voice.distress >= 0.6:
            reading.distress = max(reading.distress, 0.35)
    else:
        reading = _heuristic(text, voice)

    state["sentiment"] = reading
    trace.append({
        "agent": "sentiment",
        "summary": f"[{reading.source}] {reading.label} · valence {reading.valence:+.2f} · "
                   f"distress {reading.distress:.2f}"
                   + (" · escalate" if reading.escalate else "")
                   + (f" · voice: {', '.join(voice.flags)}" if voice.flags else ""),
        "voice": voice.summary() if voice.available else "",
    })
    return state
