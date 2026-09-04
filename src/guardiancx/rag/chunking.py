"""Parse policy markdown into retrievable, tagged chunks.

A clause carries three retrieval keys, not one:

* the **driver** it addresses (`## Driver: Health`) — the FCA vulnerability axis;
* the **journeys** it applies to (`Journeys: arrears_collections, ...`) — the
  banking situations where it is the right clause to reach for;
* the **products** it is scoped to, where it is product-specific.

A clause may also carry `Offer:` lines — the wording a handler may actually say
to the customer. Policy prose is written for staff, in the imperative and about a
third party; deriving customer-facing speech from it mechanically works for
simple clauses and produces mangled English for the rest. What a vulnerable
customer hears should be approved wording, so it is authored here alongside the
clause it comes from, where a compliance reviewer can see and sign off both.

**Offers are authored per language**, as `Offer[ar]:`, `Offer[ur]:` and so on
beside the unmarked English line. This is the difference between a multilingual
agent and a translated one. A treatment strategy that says "approved wording
only, without deviation" cannot be honoured by putting an approved English
sentence through a translation model at call time: what the customer hears would
then be wording no reviewer has ever seen, in a language no reviewer may read.
Authoring the Arabic and the Urdu here means the same compliance reviewer signs
off the same clause in every language it will be spoken in. Where a language is
missing, retrieval falls back to English and *says so* — a gap a reviewer can
close, rather than a fluent sentence nobody approved.

Driver alone retrieves plausible-sounding but wrong clauses: a bereaved customer
in a scam call and a bereaved customer in a collections call both score high on
`life_events` and need entirely different policy. The journey tag is what
separates them, and it is why retrieval in this system is journey-aware rather
than similarity-only.

Journey and product lines are metadata, not prose — they are stripped from the
chunk text so they never pollute the embedding or the quoted clause.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from ..finance.taxonomy import Journey, Product
from ..utils.language import normalise as normalise_language
from ..utils.types import Driver

_DRIVER_HEADER = re.compile(r"^##\s+Driver:\s+(.+?)\s*$", re.IGNORECASE)
_SECTION_HEADER = re.compile(r"^###\s+(\S+)\s+[—-]\s+(.+?)\s*$")
_JOURNEY_LINE = re.compile(r"^\*{0,2}Journeys?:\*{0,2}\s*(.+?)\s*$", re.IGNORECASE)
_PRODUCT_LINE = re.compile(r"^\*{0,2}Products?:\*{0,2}\s*(.+?)\s*$", re.IGNORECASE)
# `Offer:` is English; `Offer[ar]:` / `Offer[ur]:` / `Offer[fil-PH]:` carry the
# approved wording for that language. The tag is a BCP-47 primary subtag,
# optionally with a region.
_OFFER_LINE = re.compile(
    r"^\*{0,2}Offer(?:\[([A-Za-z]{2,3}(?:-[A-Za-z]{2,4})?)\])?:\*{0,2}\s*(.+?)\s*$",
    re.IGNORECASE,
)

DEFAULT_OFFER_LANGUAGE = "en"

# A clause tagged with a language this bank does not speak is a policy-authoring
# mistake, and `normalise_language` collapses it to English rather than creating
# a bucket nothing will ever read from.

_DRIVER_MAP = {
    "health": Driver.HEALTH,
    "life events": Driver.LIFE_EVENTS,
    "resilience": Driver.RESILIENCE,
    "capability": Driver.CAPABILITY,
}


def _parse_enum_list(raw: str, cls) -> list:
    out = []
    for token in re.split(r"[,;]", raw):
        token = token.strip().strip("*` ").lower().replace(" ", "_")
        if not token:
            continue
        try:
            out.append(cls(token))
        except ValueError:
            continue
    return out


class Chunk:
    def __init__(self, ref: str, title: str, driver: Optional[Driver], text: str,
                 source: str, journeys: Optional[list[Journey]] = None,
                 products: Optional[list[Product]] = None,
                 offers: Optional[list[str]] = None,
                 offers_by_language: Optional[dict[str, list[str]]] = None):
        self.ref = ref
        self.title = title
        self.driver = driver
        self.text = text
        self.source = source
        self.journeys = journeys or []
        self.products = products or []
        self.offers = offers or []
        # {"ar": [...], "ur": [...]} — English lives in `offers`, not in here.
        self.offers_by_language = offers_by_language or {}

    def offers_in(self, language: str) -> tuple[list[str], bool]:
        """Approved wording for `language`, and whether English was substituted.

        The caller needs the second half of that answer. Speaking English at a
        customer who asked for Urdu is a service failure worth recording; making
        up Urdu is a compliance breach. Falling back and flagging is neither.
        """
        code = normalise_language(language)
        if code == DEFAULT_OFFER_LANGUAGE:
            return list(self.offers), False
        authored = self.offers_by_language.get(code)
        if authored:
            return list(authored), False
        return list(self.offers), bool(self.offers)

    @property
    def embed_text(self) -> str:
        """What gets embedded.

        The journey names are included deliberately: they carry the operational
        vocabulary ("arrears collections", "bereavement estate") that customer
        utterances echo, and including them measurably improves retrieval over
        embedding the clause prose alone.
        """
        tags = " ".join(j.value.replace("_", " ") for j in self.journeys)
        return f"{self.ref} {self.title}. {self.text} {tags}".strip()

    @property
    def journey_values(self) -> list[str]:
        return [j.value for j in self.journeys]


def chunk_policy_file(path: str | Path) -> list[Chunk]:
    p = Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    chunks: list[Chunk] = []
    driver: Optional[Driver] = None
    ref = title = None
    body: list[str] = []
    journeys: list[Journey] = []
    products: list[Product] = []
    offers: list[str] = []
    offers_by_language: dict[str, list[str]] = {}

    def flush():
        nonlocal ref, title, body, journeys, products, offers, offers_by_language
        if ref and title:
            text = " ".join(l.strip() for l in body if l.strip() and l.strip() != "---")
            chunks.append(Chunk(ref, title, driver, text, p.name, journeys, products,
                                offers, offers_by_language))
        ref = title = None
        body = []
        journeys = []
        products = []
        offers = []
        offers_by_language = {}

    for line in lines:
        dm = _DRIVER_HEADER.match(line)
        if dm:
            flush()
            driver = _DRIVER_MAP.get(dm.group(1).strip().lower())
            continue
        sm = _SECTION_HEADER.match(line)
        if sm:
            flush()
            ref, title = sm.group(1).strip(), sm.group(2).strip()
            continue
        if line.startswith("## ") and not _DRIVER_HEADER.match(line):
            flush()
            driver = None
            continue
        if ref:
            jm = _JOURNEY_LINE.match(line.strip())
            if jm:
                journeys = _parse_enum_list(jm.group(1), Journey)
                continue
            pm = _PRODUCT_LINE.match(line.strip())
            if pm:
                products = _parse_enum_list(pm.group(1), Product)
                continue
            om = _OFFER_LINE.match(line.strip())
            if om:
                tag, wording = om.group(1), om.group(2).strip()
                code = normalise_language(tag)
                if code == DEFAULT_OFFER_LANGUAGE:
                    offers.append(wording)
                else:
                    offers_by_language.setdefault(code, []).append(wording)
                continue
            body.append(line)
    flush()
    return chunks


def chunk_policy_dir(directory: str | Path) -> list[Chunk]:
    out: list[Chunk] = []
    for md in sorted(Path(directory).glob("*.md")):
        out.extend(chunk_policy_file(md))
    return out
