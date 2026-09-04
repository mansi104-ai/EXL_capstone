"""The opt-out register — heard on the call, honoured before the next one.

"Every opt-out honoured" is one of the few claims in a treatment strategy that a
regulator can test directly: take the register, take the call log, and look for a
call that came after a request to stop. That test only passes if two things are
true, and this module is responsible for both.

**It has to be heard.** An opt-out is not a form the customer fills in. It is a
sentence in the middle of a call — "stop calling me", "take my number off your
list" — and often in the customer's own language rather than the bank's. A
register that can only be written by a handler remembering to tick a box is a
register that misses most of what it should hold. So the detector below runs on
every customer turn, in every language the channel supports, and the phrasings
are the ones people actually use when they have had enough.

**It has to be written immediately.** Before the call ends, not after it is
wrapped up: a call that drops between the request and the wrap-up has still had
the request made in it, and the recording proves it. Everything here writes
through to the database on the turn it happens.

A note on what counts. "Don't call me at work" is a channel restriction, not an
opt-out, and "I can't talk right now" is neither — treating either as a
withdrawal of consent would quietly strand customers who still want help. The
detector aims at unambiguous withdrawal and leaves the rest to the handler, on
the principle that a false negative here costs one more call while a false
positive costs the customer their route back to the bank.
"""
from __future__ import annotations

import re
import uuid
from typing import Optional

from ..database.db import session_scope
from ..database.models import ContactPreference
from ..utils.language import normalise as normalise_language
from ..utils.logging import get_logger

log = get_logger("outbound.optout")

# Channels a customer can opt out of. "all" is the default and the commonest:
# a customer saying "stop contacting me" has not drawn a distinction, and it is
# not the bank's place to read one in.
CHANNELS = ("all", "voice", "sms", "email", "letter")

# Unambiguous withdrawals of consent to be contacted, per language. Kept as
# phrases rather than keywords: "stop" alone appears in "stop the interest",
# which is a customer asking for help, not asking to be left alone.
_OPT_OUT_PATTERNS: dict[str, list[re.Pattern]] = {
    "en": [
        re.compile(r"\b(?:stop|quit|cease)\s+(?:calling|phoning|ringing|contacting)\s+me\b",
                   re.IGNORECASE),
        re.compile(r"\b(?:do\s*n[o']?t|don't|never)\s+(?:call|phone|ring|contact)\s+me\b",
                   re.IGNORECASE),
        re.compile(r"\btake\s+(?:me|my\s+(?:number|details))\s+off\s+(?:your|the)\s+list\b",
                   re.IGNORECASE),
        re.compile(r"\bremove\s+my\s+(?:number|details|contact)\b", re.IGNORECASE),
        re.compile(r"\b(?:opt|op)\s*[- ]?\s*out\b", re.IGNORECASE),
        re.compile(r"\bno\s+more\s+calls\b", re.IGNORECASE),
        re.compile(r"\bstop\s+contacting\b", re.IGNORECASE),
    ],
    "ar": [
        re.compile(r"لا\s+تتصل(?:وا)?\s+بي"),          # do not call me
        re.compile(r"توقف(?:وا)?\s+عن\s+الاتصال"),      # stop calling
        re.compile(r"احذف(?:وا)?\s+رقمي"),              # delete my number
        re.compile(r"لا\s+أريد\s+(?:أي\s+)?مكالمات"),   # I don't want any calls
    ],
    "ur": [
        re.compile(r"مجھے\s+کال\s+نہ\s+کریں"),          # do not call me
        re.compile(r"فون\s+کرنا\s+بند\s+کریں"),          # stop phoning
        re.compile(r"میرا\s+نمبر\s+نکال\s+دیں"),         # remove my number
        re.compile(r"مجھے\s+تنگ\s+نہ\s+کریں"),          # do not trouble me
    ],
    "hi": [
        re.compile(r"मुझे\s+कॉल\s+मत\s+कर"),
        re.compile(r"फ़?ोन\s+करना\s+बंद\s+क"),
        re.compile(r"मेरा\s+नंबर\s+हटा"),
    ],
}

# Phrases that look like an opt-out and are not. Checked first, because the
# cost of misreading one of these is that a customer asking for help is recorded
# as having refused it.
_NOT_OPT_OUT = [
    re.compile(r"\bstop\s+(?:the\s+)?(?:interest|charges|fees|letters?|"
               r"the\s+recovery|collection)\b", re.IGNORECASE),
    re.compile(r"\b(?:can'?t|cannot)\s+(?:talk|speak)\s+(?:right\s+)?now\b", re.IGNORECASE),
    re.compile(r"\b(?:call|ring|phone)\s+me\s+(?:back|later|tomorrow|after)\b", re.IGNORECASE),
    re.compile(r"\b(?:do\s*n[o']?t|don't)\s+(?:call|contact)\s+me\s+at\s+work\b",
               re.IGNORECASE),
]


def detect_opt_out(text: str, language: str = "en") -> Optional[str]:
    """Did the customer just ask not to be contacted? Returns the phrase matched.

    The customer's language is checked first and English always, because a
    customer may switch to English for the one sentence they most want
    understood.
    """
    if not text or not text.strip():
        return None

    for pattern in _NOT_OPT_OUT:
        if pattern.search(text):
            return None

    codes = [normalise_language(language), "en"]
    seen: set[str] = set()
    for code in codes:
        if code in seen:
            continue
        seen.add(code)
        for pattern in _OPT_OUT_PATTERNS.get(code, []):
            match = pattern.search(text)
            if match:
                return match.group(0)
    return None


def record_opt_out(customer_id: str, *, channel: str = "all", stated: str = "",
                   language: str = "en", source: str = "call",
                   conversation_id: str = "") -> str:
    """Write an opt-out. Returns the channel recorded.

    Appends rather than updates, so the register keeps the history of what the
    customer asked for and when. Nothing here is ever deleted.
    """
    channel = channel if channel in CHANNELS else "all"
    with session_scope() as session:
        session.add(ContactPreference(
            customer_id=customer_id, channel=channel, opted_out=True,
            stated=(stated or "")[:500], language=normalise_language(language),
            source=source, conversation_id=conversation_id,
        ))
    log.info("Opt-out recorded for %s on channel %s", customer_id, channel)
    return channel


def withdraw_opt_out(customer_id: str, *, channel: str = "all",
                     stated: str = "", source: str = "call",
                     conversation_id: str = "") -> None:
    """Record that the customer has asked to be contacted again.

    A new row, not an edit. The question a reviewer asks is never "is this
    customer opted out?" alone — it is "were they opted out on the day we
    called?", and only an append-only register can answer that.
    """
    channel = channel if channel in CHANNELS else "all"
    with session_scope() as session:
        session.add(ContactPreference(
            customer_id=customer_id, channel=channel, opted_out=False,
            stated=(stated or "")[:500], source=source,
            conversation_id=conversation_id,
        ))


def is_opted_out(customer_id: str, channel: str = "voice") -> bool:
    """Has this customer opted out of this channel?

    An "all" opt-out covers every channel. The most recent row for the specific
    channel and the most recent row for "all" are both consulted, and the later
    of the two decides — a customer who opts out of everything and then asks for
    calls back about their arrears has re-consented to calls, and a register that
    let the older blanket row win would strand them.
    """
    with session_scope() as session:
        rows = (session.query(ContactPreference)
                .filter(ContactPreference.customer_id == customer_id)
                .filter(ContactPreference.channel.in_([channel, "all"]))
                .order_by(ContactPreference.created_at.desc(),
                          ContactPreference.id.desc())
                .all())
    return bool(rows) and bool(rows[0].opted_out)


def opt_out_register(customer_id: str = "") -> list[dict]:
    """The register, newest first — for the console and for evidence."""
    with session_scope() as session:
        query = session.query(ContactPreference)
        if customer_id:
            query = query.filter(ContactPreference.customer_id == customer_id)
        rows = query.order_by(ContactPreference.created_at.desc(),
                              ContactPreference.id.desc()).limit(500).all()
        return [{
            "customer_id": r.customer_id,
            "channel": r.channel,
            "opted_out": r.opted_out,
            "stated": r.stated,
            "language": r.language,
            "source": r.source,
            "conversation_id": r.conversation_id,
            "recorded_at": r.created_at.isoformat() if r.created_at else "",
        } for r in rows]


def new_id() -> str:
    return uuid.uuid4().hex[:12]
