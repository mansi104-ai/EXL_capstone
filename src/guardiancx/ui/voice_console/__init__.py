"""The live voice console — a Streamlit component that keeps the line open.

Streamlit's built-in audio input is push-to-talk: the user presses record,
speaks, presses stop, waits. That is a form, not a phone call, and it cannot
demonstrate any of the things a real vulnerability line has to get right —
noticing that a customer paused mid-sentence, catching a spoken sort code before
it is transcribed, hearing distress in a voice.

So the call itself runs in the browser, in a small component that:

* keeps continuous recognition open (Azure Speech, falling back to the browser's
  own recogniser) and shows interim words as they are spoken;
* measures the six acoustic quantities `voice.prosody` derives its indicators
  from, frame by frame, with the Web Audio API;
* runs the end-of-utterance model locally to decide when the customer has
  actually finished, then posts the committed turn up to Python;
* plays back the synthesised reply with recognition muted, so the agent does not
  transcribe and answer its own voice.

Python stays authoritative: it re-runs the EOU model on the committed turn for
the trace and the evidence record, runs the full agent pipeline, and returns the
spoken reply. The browser copy of the model exists only because the decision to
keep listening has to be made in milliseconds, not across a network round trip.

Credentials never reach the page — the component is handed a ten-minute Azure
authorisation token, minted server-side by `SpeechService.issue_token`.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Optional

import streamlit as st

_FRONTEND = Path(__file__).parent / "frontend"
_COMPONENT = None


def _component():
    """Declare the component once per process."""
    global _COMPONENT
    if _COMPONENT is None:
        _COMPONENT = st.components.v1.declare_component(
            "guardiancx_voice_console", path=str(_FRONTEND),
        )
    return _COMPONENT


def voice_console(
    token: str = "",
    region: str = "",
    language: str = "en-GB",
    distress: float = 0.0,
    speak_audio: Optional[bytes] = None,
    speak_id: str = "",
    stop_signal: bool = False,
    key: str = "voice_console",
) -> Optional[dict[str, Any]]:
    """Render the console and return the most recent committed utterance.

    Returns ``None`` until the customer finishes a turn, then a dict with
    ``event``, ``seq``, ``text``, ``silence_ms``, ``eou_reason``,
    ``eou_completeness``, ``prosody`` and ``engine``. The ``seq`` counter is what
    callers use to tell a new utterance from the same one being replayed across
    a rerun.

    `distress` is the customer's current measured distress: it feeds the browser
    copy of the endpointing model, which extends every silence threshold for a
    caller who is struggling. `speak_audio` is the synthesised reply to play,
    paired with a `speak_id` so a rerun does not replay it.
    """
    encoded = base64.b64encode(speak_audio).decode("ascii") if speak_audio else ""
    return _component()(
        token=token,
        region=region,
        language=language,
        distress=float(distress),
        speak_audio=encoded,
        speak_id=speak_id,
        stop_signal=bool(stop_signal),
        key=key,
        default=None,
    )
