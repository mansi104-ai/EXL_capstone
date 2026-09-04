"""The languages this bank's collections book actually speaks.

The UAE retail borrower base is majority expatriate, and a collections channel
that operates in English and Arabic leaves a large share of it unable to answer a
question about its own debt. That is not a translation problem to be solved at
the end; it decides three things at the start of every call:

* **which recogniser locale** the speech engine runs — Azure continuous
  recognition accepts a bounded set of candidate locales, not "any language";
* **which neural voice** answers, because a voice reading Urdu with an English
  front end is worse than silence;
* **which approved wording** the reply composer is allowed to use, which is the
  compliance half and the one that matters most.

Everything here is deliberately small and deterministic. Detection is by script
first and function words second, so it needs no model, runs on the first
utterance, and can be read by a compliance reviewer who wants to know why the
call switched to Urdu.

**Detection proposes; the ledger and the customer decide.** An outbound call
opens in the language on the customer's record. Detection only ever offers a
correction, and a correction is applied when the evidence is strong — a customer
who answers an English greeting with two Arabic words has not necessarily asked
to continue in Arabic, and switching on that is its own kind of failure.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

DEFAULT_LANGUAGE = "en"


@dataclass(frozen=True)
class Language:
    code: str            # BCP-47 primary subtag, the key used everywhere
    english_name: str
    native_name: str
    stt_locale: str      # what Azure Speech wants for recognition
    voice: str           # the neural voice that answers
    rtl: bool = False


# The six the demo book speaks. Adding a seventh is one row here, one recogniser
# candidate, and a set of `Offer[xx]:` lines in the policy — nothing else.
LANGUAGES: dict[str, Language] = {
    "en": Language("en", "English", "English", "en-GB", "en-GB-SoniaNeural"),
    "ar": Language("ar", "Arabic", "العربية", "ar-AE", "ar-AE-FatimaNeural", rtl=True),
    "ur": Language("ur", "Urdu", "اردو", "ur-PK", "ur-PK-UzmaNeural", rtl=True),
    "hi": Language("hi", "Hindi", "हिन्दी", "hi-IN", "hi-IN-SwaraNeural"),
    "ml": Language("ml", "Malayalam", "മലയാളം", "ml-IN", "ml-IN-SobhanaNeural"),
    "fil": Language("fil", "Filipino", "Filipino", "fil-PH", "fil-PH-BlessicaNeural"),
}


def normalise(language: Optional[str]) -> str:
    """"ar-AE", "AR" and "ar" are one language as far as this system is concerned.

    Region is dropped deliberately: approved wording is signed off per language,
    not per country, and an ar-AE clause is the right thing to say to an ar-EG
    speaker. An unknown tag falls back to English rather than raising — a call in
    progress is not the place to discover an unmapped locale.
    """
    if not language:
        return DEFAULT_LANGUAGE
    code = language.strip().lower().replace("_", "-").split("-")[0]
    return code if code in LANGUAGES else DEFAULT_LANGUAGE


def get(language: Optional[str]) -> Language:
    return LANGUAGES[normalise(language)]


def stt_locale(language: Optional[str]) -> str:
    return get(language).stt_locale


def voice_for(language: Optional[str]) -> str:
    return get(language).voice


def is_rtl(language: Optional[str]) -> bool:
    return get(language).rtl


def display_name(language: Optional[str]) -> str:
    lang = get(language)
    if lang.native_name == lang.english_name:
        return lang.english_name
    return f"{lang.english_name} ({lang.native_name})"


# --------------------------------------------------------------------------- #
# Recogniser candidates
# --------------------------------------------------------------------------- #
# Azure's continuous recognition accepts at most four candidate locales for
# at-start language identification. That is a hard limit, not a tuning knob, and
# pretending otherwise produces a recogniser that silently ignores the tail of
# the list. So the candidate set is chosen per call from what is actually known
# about the customer, rather than being a fixed global list.
MAX_AUTODETECT_CANDIDATES = 4


def autodetect_locales(preferred: Optional[str] = None,
                       also: Optional[list[str]] = None) -> list[str]:
    """The recogniser candidate list for one call, most likely first.

    The customer's language on file leads. English comes second and is never
    displaced, because it is the market's lingua franca and a customer may simply
    use it — including the customer whose record says Malayalam and who has done
    business in English for twenty years. Only then do the remaining one or two
    slots go to `also` and to the commonest languages in the book.

    Ordering English ahead of `also` is deliberate and was a bug the first time
    round: with four candidates and three hinted languages, English fell off the
    end of the list, and the caller most likely to be misrecognised was the one
    who answered the phone in the language the whole market speaks.
    """
    order: list[str] = []
    for code in [preferred, "en", *(also or []), "ar", "ur", "hi"]:
        code = normalise(code) if code else None
        if code and code not in order:
            order.append(code)
    return [LANGUAGES[c].stt_locale for c in order[:MAX_AUTODETECT_CANDIDATES]]


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
# Script blocks settle most of it outright. The one genuinely hard case is Arabic
# script, which Arabic and Urdu share: the separator is a handful of letters Urdu
# added and Arabic does not use.
_URDU_ONLY = set("ٹڈڑںےھہۂۓپچژگ")
_ARABIC_RANGE = re.compile(r"[؀-ۿݐ-ݿ]")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_MALAYALAM = re.compile(r"[ഀ-ൿ]")

# Function words, for the two languages that share the Latin alphabet. Filipino
# is picked out by its particles, which are short, extremely frequent, and rare
# as standalone words in English.
#
# "hindi" is in this list and means "no" in Tagalog. It is also the English name
# of a different language on this very list, so it is scored, never decisive.
_FILIPINO_MARKERS = {
    "ako", "ang", "ng", "mga", "po", "opo", "salamat", "kailangan", "magkano",
    "bayad", "utang", "hindi", "wala", "meron", "sana", "kayo", "namin", "natin",
    "pera", "trabaho", "sahod", "pwede", "paano", "kasi", "lang", "yung",
}
_ENGLISH_MARKERS = {
    "the", "and", "is", "was", "have", "i", "you", "my", "to", "of", "for",
    "payment", "account", "loan", "please", "sorry", "can", "cannot", "not",
}

_WORD = re.compile(r"[A-Za-z']+")


@dataclass(frozen=True)
class Detection:
    language: str
    confidence: float
    basis: str           # why — shown in the trace, and readable by a reviewer

    @property
    def strong(self) -> bool:
        """Enough to switch a call that opened in another language."""
        return self.confidence >= 0.75


def detect(text: str, default: str = DEFAULT_LANGUAGE) -> Detection:
    """Identify the language of one utterance.

    Returns the default with zero confidence for anything too short to judge —
    "yes", "hmm" and a bare account number are not evidence of anything.
    """
    stripped = (text or "").strip()
    if len(stripped) < 2:
        return Detection(normalise(default), 0.0, "too short to judge")

    letters = [c for c in stripped if unicodedata.category(c).startswith("L")]
    if not letters:
        return Detection(normalise(default), 0.0, "no letters in the utterance")

    if _MALAYALAM.search(stripped):
        return Detection("ml", 0.97, "Malayalam script")
    if _DEVANAGARI.search(stripped):
        return Detection("hi", 0.97, "Devanagari script")
    if _ARABIC_RANGE.search(stripped):
        urdu_hits = sum(1 for c in stripped if c in _URDU_ONLY)
        if urdu_hits:
            return Detection("ur", 0.93,
                             f"Arabic script with {urdu_hits} Urdu-only letter(s)")
        return Detection("ar", 0.88, "Arabic script, no Urdu-only letters")

    words = [w.lower() for w in _WORD.findall(stripped)]
    if not words:
        return Detection(normalise(default), 0.0, "no Latin words to score")

    fil_hits = sum(1 for w in words if w in _FILIPINO_MARKERS)
    en_hits = sum(1 for w in words if w in _ENGLISH_MARKERS)
    if fil_hits > en_hits and fil_hits >= 2:
        share = fil_hits / len(words)
        return Detection("fil", min(0.6 + share, 0.92),
                         f"{fil_hits} Filipino function word(s) against {en_hits} English")
    if en_hits:
        return Detection("en", min(0.55 + 0.1 * en_hits, 0.9),
                         f"{en_hits} English function word(s)")
    return Detection(normalise(default), 0.3, "Latin script, no decisive markers")


def resolve_call_language(on_file: Optional[str], utterance: str,
                          current: Optional[str] = None) -> tuple[str, str]:
    """The language this call should continue in, and why.

    The record is the starting point; a strong detection overrides it. A weak
    detection does not, because switching a distressed customer into the wrong
    language mid-sentence is worse than continuing in a language they can at
    least get by in.
    """
    settled = normalise(current or on_file)
    found = detect(utterance, default=settled)
    if found.language != settled and found.strong:
        return found.language, f"switched to {display_name(found.language)}: {found.basis}"
    return settled, f"continuing in {display_name(settled)}"
