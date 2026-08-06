"""Generate synthetic labeled data + sample conversations for the VCA.

Outputs:
  data/labeled/train.jsonl, val.jsonl  -- multi-label utterances for the classifier
  data/transcripts/conversation_*.json -- full sample conversations for the demo

Everything here is synthetic. No real customer data is used.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELED_DIR = ROOT / "data" / "labeled"
TRANSCRIPT_DIR = ROOT / "data" / "transcripts"

DRIVERS = ["health", "life_events", "resilience", "capability"]

# Customer phrases that signal each driver. Templates use {x} slots filled below.
DRIVER_PHRASES: dict[str, list[str]] = {
    "health": [
        "I've just been diagnosed with cancer and I'm starting chemo next week.",
        "I'm registered disabled and it's hard for me to get to a branch.",
        "My anxiety has been really bad lately, I'm honestly not coping.",
        "I've got a terminal illness so I'm trying to sort my affairs out.",
        "I'm in and out of hospital at the moment with my heart condition.",
        "I struggle with my memory since the stroke, I forget things.",
        "I have depression and some days I can't even open the post.",
        "I'm on strong medication and it makes it hard to concentrate.",
    ],
    "life_events": [
        "My husband passed away last month and I'm dealing with his accounts.",
        "I lost my job on Friday and I don't know how I'll manage.",
        "We're going through a divorce and need to split the joint account.",
        "My mum died recently and I'm the executor of her estate.",
        "I've had to stop working to care for my disabled son full time.",
        "My hours were cut and my income has basically halved.",
        "My partner and I have separated so I've moved out.",
        "I was made redundant and the redundancy pay has run out.",
    ],
    "resilience": [
        "I genuinely can't afford this payment, I'm choosing between heating and eating.",
        "I've fallen behind on everything and the debts are piling up.",
        "I've got no savings at all, one unexpected bill would finish me.",
        "I'm borrowing on one card to pay another, it's a mess.",
        "If you take that payment I won't have money for food this week.",
        "I'm really struggling financially, I don't know where to turn.",
        "I owe money to five different lenders and I can't keep up.",
        "There's nothing left after the bills, I'm living hand to mouth.",
    ],
    "capability": [
        "I don't really understand all these financial terms, sorry.",
        "I'm not good with computers, can I do this over the phone instead?",
        "English isn't my first language so please speak slowly.",
        "My daughter usually helps me with this sort of thing.",
        "I've got power of attorney for my father, I'm calling on his behalf.",
        "Can you explain that in plain English? I'm a bit lost.",
        "I find all the online stuff really confusing.",
        "I'd like my son to be able to speak to you about my account.",
    ],
}

# Neutral customer utterances (no driver).
NEUTRAL = [
    "Hi, I'd like to check the balance on my current account.",
    "Can you tell me what my interest rate is?",
    "I want to set up a standing order to my landlord.",
    "What time do your branches close on a Saturday?",
    "I'd like to order a new debit card please.",
    "Can I increase my overdraft limit?",
    "I'm calling to update my address, I've moved.",
    "How do I set up the mobile app?",
    "I think there's a duplicate charge on my statement.",
    "Can you confirm my last three transactions?",
    "I'd like to book an appointment with a mortgage adviser.",
    "What's the exchange rate for euros today?",
]

HANDLER_LINES = [
    "Thanks for calling Acme, how can I help you today?",
    "Of course, let me pull up your account.",
    "I understand. Let me see what options we have.",
    "Thank you for letting me know.",
    "Is there anything else I can help you with?",
    "Let me just confirm a couple of security details.",
]


def _labels(active: list[str]) -> dict[str, int]:
    return {d: (1 if d in active else 0) for d in DRIVERS}


def build_labeled_rows(seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []

    # Single-driver examples.
    for driver, phrases in DRIVER_PHRASES.items():
        for text in phrases:
            rows.append({"text": text, "labels": _labels([driver])})

    # Multi-driver examples (co-occurring vulnerabilities are common).
    combos = [
        ("life_events", "resilience"),
        ("health", "capability"),
        ("life_events", "health"),
        ("resilience", "capability"),
    ]
    for a, b in combos:
        for _ in range(6):
            text = rng.choice(DRIVER_PHRASES[a]) + " " + rng.choice(DRIVER_PHRASES[b])
            rows.append({"text": text, "labels": _labels([a, b])})

    # Neutral examples (no driver) -- important for precision.
    for text in NEUTRAL:
        rows.append({"text": text, "labels": _labels([])})
    for _ in range(12):
        text = rng.choice(NEUTRAL) + " " + rng.choice(NEUTRAL)
        rows.append({"text": text, "labels": _labels([])})

    rng.shuffle(rows)
    return rows


def build_sample_conversations() -> list[dict]:
    """A few full conversations used by the demo / UI."""
    return [
        {
            "conversation_id": "CONV-001",
            "description": "Bereavement + financial distress",
            "turns": [
                {"speaker": "handler", "text": "Thanks for calling Acme, how can I help today?"},
                {"speaker": "customer", "text": "My husband passed away last month and I'm dealing with his accounts."},
                {"speaker": "handler", "text": "I'm so sorry for your loss. Let me help you with that."},
                {"speaker": "customer", "text": "Thank you. The thing is, money is really tight now, I genuinely can't afford the loan payment this month."},
                {"speaker": "handler", "text": "I understand. Let me look at what we can do."},
                {"speaker": "customer", "text": "I've got no savings at all, one unexpected bill would finish me."},
            ],
        },
        {
            "conversation_id": "CONV-002",
            "description": "Health + capability",
            "turns": [
                {"speaker": "handler", "text": "Hi, you're through to Acme, how can I help?"},
                {"speaker": "customer", "text": "I struggle with my memory since the stroke, I forget things."},
                {"speaker": "handler", "text": "No problem, we can take it slowly."},
                {"speaker": "customer", "text": "My daughter usually helps me with this sort of thing."},
                {"speaker": "customer", "text": "Can you explain that in plain English? I'm a bit lost."},
            ],
        },
        {
            "conversation_id": "CONV-003",
            "description": "Neutral (no vulnerability)",
            "turns": [
                {"speaker": "handler", "text": "Thanks for calling Acme, how can I help you today?"},
                {"speaker": "customer", "text": "Hi, I'd like to check the balance on my current account."},
                {"speaker": "handler", "text": "Of course, let me confirm a couple of security details."},
                {"speaker": "customer", "text": "Sure. Also can I order a new debit card please?"},
            ],
        },
        {
            "conversation_id": "CONV-004",
            "description": "Job loss / income shock",
            "turns": [
                {"speaker": "handler", "text": "You're through to Acme, how can I help?"},
                {"speaker": "customer", "text": "I lost my job on Friday and I don't know how I'll manage."},
                {"speaker": "handler", "text": "I'm sorry to hear that. Let's look at your options."},
                {"speaker": "customer", "text": "If you take that payment I won't have money for food this week."},
            ],
        },
    ]


def main() -> None:
    LABELED_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    rows = build_labeled_rows()
    split = int(len(rows) * 0.8)
    train, val = rows[:split], rows[split:]

    with open(LABELED_DIR / "train.jsonl", "w", encoding="utf-8") as fh:
        for r in train:
            fh.write(json.dumps(r) + "\n")
    with open(LABELED_DIR / "val.jsonl", "w", encoding="utf-8") as fh:
        for r in val:
            fh.write(json.dumps(r) + "\n")

    for conv in build_sample_conversations():
        path = TRANSCRIPT_DIR / f"{conv['conversation_id']}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(conv, fh, indent=2)

    print(f"Wrote {len(train)} train / {len(val)} val labeled rows to {LABELED_DIR}")
    print(f"Wrote {len(build_sample_conversations())} sample conversations to {TRANSCRIPT_DIR}")


if __name__ == "__main__":
    main()
