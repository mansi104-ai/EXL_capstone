"""The finance domain taxonomy.

A general-purpose vulnerability classifier can tell you *that* a customer is
struggling. It cannot tell you that a customer three EMIs behind on a home loan,
choosing between the electricity bill and the instalment, is owed forbearance
under the RBI Fair Practices Code — and that a recovery call to them after 7pm,
or an offer of a top-up loan, would breach it.

That gap is the niche. This module encodes it in four parts:

* **Product** — what the customer holds, because the obligations differ. A
  credit card sits under the RBI Master Direction on Credit and Debit Cards; a
  savings account under the Master Circular on Customer Service; a home loan
  under the Fair Practices Code. The same disclosure means different things
  against each.
* **Journey** — what the conversation is *for*. Harm concentrates in a handful
  of journeys (collections, bereavement, scam, forbearance), and the journey is
  a far better retrieval key than the raw utterance.
* **StressIndicator** — the finance-specific evidence of detriment, which is not
  the same thing as the four FCA vulnerability drivers. A customer can be highly
  capable and financially literate and still be in arrears; a customer can be
  bereaved with no financial stress at all. The two axes are independent, and
  the intersection is where the firm must act.
* **The regulatory map** — journey → the rules that bite, so that every piece of
  guidance can be traced to an obligation rather than to a tone of voice.

A note on the four drivers. Health, life events, resilience and capability are
not an RBI construct — they come from the UK regulator's vulnerability guidance,
which is the most fully worked-out articulation of the idea anywhere. Pan Indian
Bank adopts them as **its own** framework for identifying customers at risk, and
grounds the *obligations* that follow in Indian instruments: the RBI Charter of
Customer Rights, the Fair Practices Code, the Master Circular on Customer
Service, and the Integrated Ombudsman Scheme. That split is deliberate and worth
being explicit about: the taxonomy is the bank's, the duties are the regulator's.

Everything here is deterministic and inspectable. The LLM classifier in
`agents/financial_context_agent.py` proposes; this taxonomy defines the
vocabulary it must answer in, and provides the keyword fallback when no model is
configured.
"""
from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Products
# --------------------------------------------------------------------------- #
class Product(str, Enum):
    CURRENT_ACCOUNT = "current_account"
    SAVINGS = "savings"
    CREDIT_CARD = "credit_card"
    PERSONAL_LOAN = "personal_loan"
    MORTGAGE = "mortgage"
    OVERDRAFT = "overdraft"
    CAR_FINANCE = "car_finance"
    PENSION = "pension"
    INSURANCE = "insurance"
    UNKNOWN = "unknown"


PRODUCT_LABELS: dict[Product, str] = {
    Product.CURRENT_ACCOUNT: "Current account",
    Product.SAVINGS: "Savings",
    Product.CREDIT_CARD: "Credit card",
    Product.PERSONAL_LOAN: "Personal loan",
    Product.MORTGAGE: "Home loan",
    Product.OVERDRAFT: "Overdraft",
    Product.CAR_FINANCE: "Vehicle loan",
    Product.PENSION: "Pension",
    Product.INSURANCE: "Insurance",
    Product.UNKNOWN: "Not yet identified",
}

# The RBI instrument that governs each product, shown in the guidance panel so a
# handler can see which rulebook the advice is grounded in.
PRODUCT_SOURCEBOOK: dict[Product, str] = {
    Product.CURRENT_ACCOUNT: "Master Circular on Customer Service",
    Product.SAVINGS: "Master Circular on Customer Service",
    Product.CREDIT_CARD: "MD on Credit & Debit Cards",
    Product.PERSONAL_LOAN: "Fair Practices Code",
    Product.MORTGAGE: "Fair Practices Code (home loans)",
    Product.OVERDRAFT: "Master Circular on Customer Service",
    Product.CAR_FINANCE: "Fair Practices Code",
    Product.PENSION: "Master Circular on Customer Service",
    Product.INSURANCE: "IRDAI Protection of Policyholders' Interests",
    Product.UNKNOWN: "—",
}


# --------------------------------------------------------------------------- #
# Journeys — where harm concentrates
# --------------------------------------------------------------------------- #
class Journey(str, Enum):
    ARREARS_COLLECTIONS = "arrears_collections"
    FORBEARANCE_REQUEST = "forbearance_request"
    BEREAVEMENT_ESTATE = "bereavement_estate"
    FRAUD_SCAM = "fraud_scam"
    AFFORDABILITY_SHOCK = "affordability_shock"
    GAMBLING_HARM = "gambling_harm"
    THIRD_PARTY_ACCESS = "third_party_access"
    PRODUCT_SALE = "product_sale"
    COMPLAINT = "complaint"
    GENERAL_SERVICING = "general_servicing"


JOURNEY_LABELS: dict[Journey, str] = {
    Journey.ARREARS_COLLECTIONS: "Arrears & collections",
    Journey.FORBEARANCE_REQUEST: "Forbearance request",
    Journey.BEREAVEMENT_ESTATE: "Bereavement & estate",
    Journey.FRAUD_SCAM: "Fraud & scam",
    Journey.AFFORDABILITY_SHOCK: "Affordability / income shock",
    Journey.GAMBLING_HARM: "Gambling harm",
    Journey.THIRD_PARTY_ACCESS: "Third-party access & POA",
    Journey.PRODUCT_SALE: "Product sale or switch",
    Journey.COMPLAINT: "Complaint",
    Journey.GENERAL_SERVICING: "General servicing",
}

# Journeys where getting it wrong causes foreseeable harm — these carry a risk
# floor regardless of what the vulnerability classifier scores.
HIGH_HARM_JOURNEYS: set[Journey] = {
    Journey.ARREARS_COLLECTIONS,
    Journey.FORBEARANCE_REQUEST,
    Journey.BEREAVEMENT_ESTATE,
    Journey.FRAUD_SCAM,
    Journey.AFFORDABILITY_SHOCK,
    Journey.GAMBLING_HARM,
}


# --------------------------------------------------------------------------- #
# Financial stress indicators — the detriment axis
# --------------------------------------------------------------------------- #
class StressIndicator(str, Enum):
    MISSED_PAYMENT = "missed_payment"
    ARREARS = "arrears"
    ESSENTIAL_SPEND_CONFLICT = "essential_spend_conflict"
    NO_SAVINGS_BUFFER = "no_savings_buffer"
    OVER_INDEBTEDNESS = "over_indebtedness"
    BENEFIT_RELIANCE = "benefit_reliance"
    INCOME_SHOCK = "income_shock"
    HIGH_COST_CREDIT = "high_cost_credit_reliance"
    SCAM_EXPOSURE = "scam_exposure"
    GAMBLING_SPEND = "gambling_spend"


STRESS_LABELS: dict[StressIndicator, str] = {
    StressIndicator.MISSED_PAYMENT: "Missed payment",
    StressIndicator.ARREARS: "In arrears",
    StressIndicator.ESSENTIAL_SPEND_CONFLICT: "Choosing between essentials",
    StressIndicator.NO_SAVINGS_BUFFER: "No savings buffer",
    StressIndicator.OVER_INDEBTEDNESS: "Over-indebted",
    StressIndicator.BENEFIT_RELIANCE: "Benefit reliance",
    StressIndicator.INCOME_SHOCK: "Income shock",
    StressIndicator.HIGH_COST_CREDIT: "High-cost credit reliance",
    StressIndicator.SCAM_EXPOSURE: "Scam exposure",
    StressIndicator.GAMBLING_SPEND: "Gambling spend",
}

# Indicators that mean the customer is already suffering detriment, not merely
# at risk of it. Their presence forces a human into the loop.
ACUTE_STRESS: set[StressIndicator] = {
    StressIndicator.ESSENTIAL_SPEND_CONFLICT,
    StressIndicator.ARREARS,
    StressIndicator.SCAM_EXPOSURE,
    StressIndicator.GAMBLING_SPEND,
}


# --------------------------------------------------------------------------- #
# Regulatory map — journey → the obligations that bite
# --------------------------------------------------------------------------- #
REGULATORY_MAP: dict[Journey, list[str]] = {
    Journey.ARREARS_COLLECTIONS: [
        "RBI Fair Practices Code — no coercive recovery; contact only between "
        "08:00 and 19:00",
        "RBI Master Direction on Outsourcing — conduct of recovery agents",
        "RBI Charter of Customer Rights — Right to Fair Treatment",
    ],
    Journey.FORBEARANCE_REQUEST: [
        "RBI Fair Practices Code — restructuring and relief where hardship is genuine",
        "RBI Resolution Framework for stressed personal loans",
    ],
    Journey.BEREAVEMENT_ESTATE: [
        "RBI settlement of claims of deceased depositors — simplified procedure, "
        "settlement within 15 days of a complete claim",
        "RBI Master Circular on Customer Service — nomination and survivorship",
    ],
    Journey.FRAUD_SCAM: [
        "RBI limiting customer liability in unauthorised electronic transactions "
        "(2017) — zero liability where the customer reports promptly",
        "RBI Charter of Customer Rights — Right to Privacy",
    ],
    Journey.AFFORDABILITY_SHOCK: [
        "RBI Fair Practices Code — assess repayment capacity before lending",
        "RBI Charter of Customer Rights — Right to Suitability",
    ],
    Journey.GAMBLING_HARM: [
        "RBI Charter of Customer Rights — Right to Suitability",
        "Pan Indian Bank vulnerable-customer framework — addiction as a health driver",
    ],
    Journey.THIRD_PARTY_ACCESS: [
        "RBI Master Circular on Customer Service — mandates, nomination and "
        "operation by a third party",
        "Rights of Persons with Disabilities Act 2016 — reasonable accommodation",
    ],
    Journey.PRODUCT_SALE: [
        "RBI Charter of Customer Rights — Right to Suitability",
        "RBI Fair Practices Code — no mis-selling or bundled products",
    ],
    Journey.COMPLAINT: [
        "RBI Integrated Ombudsman Scheme 2021 — internal redress within 30 days",
        "RBI Charter of Customer Rights — Right to Grievance Redress",
    ],
    Journey.GENERAL_SERVICING: [
        "RBI Master Circular on Customer Service",
        "RBI Charter of Customer Rights — Right to Fair Treatment",
    ],
}

# Actions that must never be offered on a given journey. The guidance agent is
# told these, and the compliance agent checks the drafted advice against them —
# a deterministic backstop that does not depend on the model behaving.
PROHIBITED_ACTIONS: dict[Journey, list[str]] = {
    Journey.GAMBLING_HARM: [
        "offering additional credit, a limit increase, or a new credit product",
    ],
    Journey.AFFORDABILITY_SHOCK: [
        "offering a top-up loan or further borrowing as the first remedy",
        "pressing for a payment that would leave essential bills unpaid",
    ],
    Journey.ARREARS_COLLECTIONS: [
        "threatening legal action or recovery agents while a relief review is open",
        "pressing for a payment that would leave essential bills unpaid",
        "calling outside 08:00–19:00, or contacting the customer's relatives or "
        "employer about the debt",
    ],
    Journey.BEREAVEMENT_ESTATE: [
        "requiring the customer to repeat the bereavement to another team",
        "applying charges or interest to the account while the estate is settled",
    ],
    Journey.FRAUD_SCAM: [
        "asking the customer to move money to a 'safe account'",
        "blaming the customer for falling victim to the scam",
    ],
}


# --------------------------------------------------------------------------- #
# The classified context that flows through the graph
# --------------------------------------------------------------------------- #
class FinancialContext(BaseModel):
    """What this conversation is about, in the firm's own terms."""

    product: Product = Product.UNKNOWN
    journey: Journey = Journey.GENERAL_SERVICING
    stress_indicators: list[StressIndicator] = Field(default_factory=list)
    arrears_months: int = 0        # 0 = none stated
    monetary_amounts: list[str] = Field(default_factory=list)
    rationale: str = ""
    source: str = "heuristic"      # "anthropic" | "openrouter" | "heuristic"

    @property
    def high_harm(self) -> bool:
        return self.journey in HIGH_HARM_JOURNEYS

    @property
    def acute(self) -> bool:
        """Detriment is already happening, not merely foreseeable."""
        return bool(set(self.stress_indicators) & ACUTE_STRESS) or self.arrears_months >= 2

    @property
    def obligations(self) -> list[str]:
        return REGULATORY_MAP.get(self.journey, [])

    @property
    def prohibited(self) -> list[str]:
        return PROHIBITED_ACTIONS.get(self.journey, [])

    @property
    def sourcebook(self) -> str:
        return PRODUCT_SOURCEBOOK.get(self.product, "—")

    def summary(self) -> str:
        bits = [f"{JOURNEY_LABELS[self.journey]}"]
        if self.product is not Product.UNKNOWN:
            bits.append(PRODUCT_LABELS[self.product])
        if self.stress_indicators:
            bits.append(", ".join(STRESS_LABELS[s] for s in self.stress_indicators))
        return " · ".join(bits)


# --------------------------------------------------------------------------- #
# Keyword fallback — used when no LLM is configured
#
# Deliberately phrase-level rather than word-level: "behind" alone means little,
# "behind on my payments" is an arrears disclosure.
# --------------------------------------------------------------------------- #
_PRODUCT_PATTERNS: list[tuple[Product, list[str]]] = [
    (Product.MORTGAGE, ["home loan", "housing loan", "mortgage", "griha"]),
    (Product.CREDIT_CARD, ["credit card", "my card", "card balance", "minimum payment"]),
    (Product.PERSONAL_LOAN, ["personal loan", "the loan", "loan payment", "loan agreement"]),
    (Product.OVERDRAFT, ["overdraft", "overdrawn"]),
    (Product.CAR_FINANCE, ["vehicle loan", "car loan", "two wheeler loan", "bike loan"]),
    (Product.SAVINGS, ["savings account", "fixed deposit", "recurring deposit",
                       "my savings", "fd"]),
    (Product.PENSION, ["pension", "annuity", "drawdown"]),
    (Product.INSURANCE, ["insurance", "policy claim", "life cover"]),
    (Product.CURRENT_ACCOUNT, ["current account", "savings account", "auto debit",
                               "standing instruction", "my account", "balance", "upi"]),
]

_JOURNEY_PATTERNS: list[tuple[Journey, list[str]]] = [
    (Journey.FRAUD_SCAM, ["scam", "scammed", "fraud", "fraudulent", "unauthorised",
                          "didn't make this payment", "safe account", "impersonat",
                          "someone took", "phishing"]),
    (Journey.BEREAVEMENT_ESTATE, ["passed away", "died", "death certificate", "probate",
                                  "executor", "estate", "funeral", "bereave", "widow"]),
    (Journey.GAMBLING_HARM, ["gambling", "betting", "casino", "bookies", "gamble"]),
    (Journey.THIRD_PARTY_ACCESS, ["power of attorney", "attorney", "carer", "trusted person",
                                  "third party", "my daughter helps", "my son helps",
                                  "helps me with"]),
    (Journey.FORBEARANCE_REQUEST, ["moratorium", "payment holiday", "pause my emi",
                                   "reduce my emi", "restructure", "payment plan",
                                   "arrangement to pay", "waive the interest",
                                   "reschedule", "more time to pay"]),
    (Journey.ARREARS_COLLECTIONS, ["behind on", "behind with", "missed a payment",
                                   "missed payments", "missed emi", "in arrears", "arrears",
                                   "overdue", "default notice", "recovery agent",
                                   "collection agent", "chasing me", "keep calling me"]),
    (Journey.AFFORDABILITY_SHOCK, ["lost my job", "made redundant", "redundant", "can't afford",
                                   "cannot afford", "hours were cut", "no income",
                                   "money for food", "struggling to pay", "money is tight",
                                   "money's tight"]),
    (Journey.COMPLAINT, ["complain", "complaint", "ombudsman", "unacceptable"]),
    (Journey.PRODUCT_SALE, ["apply for", "new card", "increase my limit", "borrow more",
                            "top-up loan", "top up loan", "switch to", "pre-approved"]),
]

# Regex rather than substring, because people do not speak in canonical phrases.
# "missed two payments", "three payments behind", "behind on my car finance" all
# describe arrears and none of them contain the phrase "in arrears".
_STRESS_PATTERNS: list[tuple[StressIndicator, str]] = [
    (StressIndicator.MISSED_PAYMENT,
     r"missed\s+(?:\w+\s+){0,2}(?:payments?|emis?)|didn'?t\s+pay"
     r"|late\s+(?:with\s+)?(?:payments?|emis?)"
     r"|(?:payment|emi|cheque)\s+(?:didn'?t|bounced|failed|returned)"),
    (StressIndicator.ARREARS,
     r"\barrears\b|\boverdue\b|behind\s+(?:on|with)\b|(?:payments?|emis?)\s+behind"
     r"|default\s+notice|fallen\s+behind|\bnpa\b|recovery\s+agent"),
    (StressIndicator.ESSENTIAL_SPEND_CONFLICT,
     r"money\s+for\s+food|can'?t\s+(?:feed|afford\s+to\s+eat)"
     r"|choos(?:e|ing)\s+between|keep\s+the\s+lights\s+on"
     r"|(?:rent|emi|electric\w*|gas|school\s+fees|medical)\s+or\s+(?:the\s+)?\w+"
     r"|no\s+money\s+for|ration"),
    (StressIndicator.NO_SAVINGS_BUFFER,
     r"no\s+savings|nothing\s+(?:put\s+by|saved|left)|no\s+buffer|nothing\s+to\s+fall\s+back"),
    (StressIndicator.OVER_INDEBTEDNESS,
     r"too\s+much\s+debt|\bdebts\b|multiple\s+(?:loans|cards)|maxed\s+(?:out|the)"
     r"|robbing\s+peter"),
    (StressIndicator.BENEFIT_RELIANCE,
     r"\bpension\b|widow\s+pension|ration\s+card|\bmnrega\b|government\s+scheme"
     r"|disability\s+(?:pension|allowance)|\bsubsidy\b"),
    (StressIndicator.INCOME_SHOCK,
     r"lost\s+my\s+job|made\s+redundant|\bredundant\b|hours\s+(?:were\s+)?cut"
     r"|sick\s+pay|no\s+income|income\s+(?:has\s+)?(?:dropped|fallen)|laid\s+off"),
    (StressIndicator.HIGH_COST_CREDIT,
     r"money\s?lender|sahukar|chit\s+fund|gold\s+loan|instant\s+loan\s+app"
     r"|buy\s+now\s+pay\s+later|loan\s+app"),
    (StressIndicator.SCAM_EXPOSURE,
     r"scam\w*|fraud\w*|safe\s+account|someone\s+took|phishing|didn'?t\s+authorise"),
    (StressIndicator.GAMBLING_SPEND,
     r"gambl\w*|betting|\bcasino\b|bookies|\bbet\b"),
]
_STRESS_RE = [(ind, re.compile(pattern, re.IGNORECASE))
              for ind, pattern in _STRESS_PATTERNS]

_COUNT = r"(one|two|three|four|five|six|seven|eight|nine|ten|\d{1,2})"
# Both orders people actually use: "three months behind" and "missed two payments".
_ARREARS_MONTHS = [
    re.compile(rf"\b{_COUNT}\s+(?:month|months|payment|payments)\s+"
               r"(?:behind|in\s+arrears|late|overdue|missed)\b", re.IGNORECASE),
    re.compile(rf"\bmissed\s+{_COUNT}\s+(?:month|months|payment|payments)\b",
               re.IGNORECASE),
]
_WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}

_MONEY = re.compile(r"(?:₹|rs\.?|inr)\s?\d[\d,]*(?:\.\d{1,2})?"
                    r"|\d[\d,]*(?:\.\d+)?\s*(?:lakh|lakhs|crore|crores|thousand)",
                    re.IGNORECASE)


def _matches(low: str, phrases: list[str]) -> int:
    return sum(1 for p in phrases if p in low)


def classify_heuristic(text: str) -> FinancialContext:
    """Deterministic finance-domain classification. Always available."""
    low = text.lower()

    product = Product.UNKNOWN
    best = 0
    for prod, phrases in _PRODUCT_PATTERNS:
        hits = _matches(low, phrases)
        if hits > best:
            product, best = prod, hits

    # Journey patterns are ordered by harm severity; the first with any hit wins,
    # so a scam disclosure inside an arrears call is still handled as a scam.
    journey = Journey.GENERAL_SERVICING
    for jrn, phrases in _JOURNEY_PATTERNS:
        if _matches(low, phrases):
            journey = jrn
            break

    stress = [ind for ind, pattern in _STRESS_RE if pattern.search(low)]

    months = 0
    for pattern in _ARREARS_MONTHS:
        m = pattern.search(low)
        if m:
            token = m.group(1)
            months = _WORD_NUMBERS.get(token, 0) or (int(token) if token.isdigit() else 0)
            break

    return FinancialContext(
        product=product,
        journey=journey,
        stress_indicators=stress,
        arrears_months=months,
        monetary_amounts=_MONEY.findall(text)[:4],
        rationale="Phrase-match over the finance taxonomy (no LLM configured).",
        source="heuristic",
    )
