# Phase 5 — Finance niche & the real call

> **Status: complete.** 103 tests passing · RAG hit-rate@3 100% · MRR 0.95 ·
> no new third-party dependency.

Final submission phase. Two goals, held together:

1. **Narrow the niche.** GuardianCX stops being "a vulnerability classifier" and
   becomes **vulnerable-customer protection for UK retail banking & consumer
   credit** — product-aware, journey-aware, and mapped to the specific rules that
   bite (FCA FG21/1, Consumer Duty PRIN 2A, CONC 7 forbearance, Breathing Space,
   APP-fraud reimbursement, BCOBS, MCOB).
2. **Make the Live Conversation Monitor a real call.** Speech in → streamed
   reasoning → speech out, with the vulnerability pipeline running *inside* the
   call rather than after it.

The problem statement does not move: a **multi-agent, advisory** system that
detects vulnerable customers, grounds guidance in policy, keeps a human in the
loop, and evidences fair treatment.

---

## Track 1 — Finance niche

| | Task | Delivered |
|---|------|-----------|
| 1.1 | Finance domain taxonomy | `finance/taxonomy.py` — 10 products, 10 journeys, 10 stress indicators, journey→regulation map, per-journey prohibited actions |
| 1.2 | **Financial Context Agent** (new) | classifies product · journey · financial stress · arrears months; LLM with a taxonomy-bound schema, deterministic fallback |
| 1.3 | Finance-specific policy corpus | `data/policies/banking_journeys.md` — 13 new clauses (CONC 7.3, Breathing Space, APP fraud, estate, POA, gambling, sell-into-vulnerability, DISP); all 34 clauses journey-tagged |
| 1.4 | Journey-aware retrieval | driver filter admits cross-cutting clauses; journey re-ranking; high-harm journeys retrieve with no driver triggered |
| 1.5 | Finance-tuned prompts | shared banking preamble; detection, guidance, sentiment and handler prompts all carry the sector |
| 1.6 | **Prohibited-action guardrail** (new) | deterministic, journey-scoped block on harmful advice; ignores negated mentions and verbatim policy quotations |

## Track 2 — The live call

| | Task | Delivered |
|---|------|-----------|
| 2.1 | Azure **TTS** + browser token | `speech.synthesize()` (SSML, style chosen by customer emotion), `speech.issue_token()` (10-minute credential; the key never reaches the page) |
| 2.2 | **EOU model** — semantic endpointing | `voice/endpointing.py` — scores syntactic completeness, derives an adaptive silence window (quadratic), extends it for a distressed caller; optional LLM adjudication for the ambiguous band |
| 2.3 | **Prosody** from the audio | `voice/prosody.py` — six measurements → agitation, tremor, hesitancy, distress; standard-library DSP, NCCF pitch |
| 2.4 | **Live PII** on the transcript | `voice/live_pii.py` — spoken figures ("oh nine, double two"), NI numbers, postcodes, dates of birth; fires pre-emptively on the *announcement* |
| 2.5 | Voice console component | continuous Azure recognition (Web Speech fallback), interim transcript, in-browser EOU + prosody, TTS playback with recognition muted |

## Track 3 — Streaming & interface

| | Task | Delivered |
|---|------|-----------|
| 3.1 | **Streaming LLM** | `LLMClient.stream_text()` for Anthropic and OpenRouter; `claude-opus-5`, adaptive thinking, per-call-site effort (low for the spoken reply, high for guidance) |
| 3.2 | Live monitor rebuild | `ui/live_monitor.py` — Call · Chat · Library over one pipeline, plus the live signal rail |
| 3.3 | **Save live conversations** | `ConversationRecord` table + repository; live sessions join the library, timeline and analytics |
| 3.4 | **Sentiment Agent** (new) | fuses words with voice; asymmetric by design — acoustics corroborate, never conclude |

## Track 4 — Proof

| | Task | Delivered |
|---|------|-----------|
| 4.1 | Tests | 103 passing: `test_voice.py` (32), `test_reply.py` (27), `test_finance.py` (26), `test_live_monitor.py` (6 headless render tests), plus the existing suite |
| 4.2 | Evaluation | harness scores the set with **and without** journey re-ranking, so the finance layer's contribution is measured; labels are sets where several clauses are genuinely correct |
| 4.3 | Docs | `OVERVIEW.md` and `README.md` rewritten to the niche and the nine-agent graph |

## Track 5 — Saying it so the customer understands

| | Task | Delivered |
|---|------|-----------|
| 5.1 | **Customer-facing reply composer** | `agents/reply.py` — approved `Offer:` wording authored per clause and used verbatim; derivation as the fallback; openers led by situation not driver score; at most two offers, then hand the turn back |
| 5.2 | **Clarity guardrail** (8th) | flags policy voice, jargon and rulebook codes, empty offers, and sentences too long to follow when spoken — the only guardrail protecting the customer rather than the firm |
| 5.3 | Distress as a retrieval trigger | a crisis disclosure no longer needs to match a keyword to reach the support policy |

---

## Defects the work surfaced

Recorded because each was found by a test or an eval run rather than by reading:

* **Pitch was read systematically low.** The autocorrelation normalised by overlap
  *length*, which grows with lag and biases the search toward long lags. Replaced
  with normalised cross-correlation over both segments' energy. Fixed in Python
  and in the browser copy.
* **A finished sentence waited 1.26 s.** The silence window interpolated linearly;
  squaring the remaining uncertainty makes the loop decisive when it is confident
  and patient when it is not.
* **Spoken-number redaction destroyed NI numbers and dates of birth** by consuming
  their digits before the specific patterns ran. Written forms now run first.
* **The prohibited-action guardrail blocked correct advice.** Policy clauses name
  forbidden actions in order to forbid them; negation detection plus verbatim-
  quotation exclusion fixed it without weakening the check.
* **Cross-cutting policy clauses were unreachable.** Clauses with no driver — the
  ban on selling into vulnerability, complaint handling, recording duties — were
  excluded by every driver-filtered query. Driver filters now admit them, which
  took hit-rate@3 from 0.969 to 1.000.
* **The handler was reading staff instructions aloud.** The reply quoted policy
  verbatim — "reassure the customer they will not need to repeat the bereavement
  disclosure" — spoken at a widow, and truncated mid-word.
* **A crisis disclosure was answered with account servicing.** "I feel hopeless,
  I don't know how I'll carry on" matched no driver keyword and no journey
  pattern, so nothing was retrieved and the reply was "let me bring up your
  account". Measured distress is now a retrieval trigger in its own right, and
  the crisis clause reaches the customer with the Samaritans number.
* **The plain-English lexicon assumed the deceased was a husband**, and told
  every bereaved customer so, whoever they had lost.
