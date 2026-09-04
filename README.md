# 🛡️ GuardianCX

**Governed collections and vulnerable-customer protection for UAE retail banking.**

GuardianCX runs **scheduled, routine collections calls that follow an approved
treatment strategy without deviation**, in the language the customer actually
speaks, and hands the call to a person the moment it stops being routine.

A nine-agent LangGraph pipeline classifies the banking situation (product,
journey, financial-stress indicators) and the customer's vulnerability across the
four drivers, retrieves the firm's prescribed adaptation via journey-aware RAG,
enforces a ten-guardrail stack, requires human approval for high-risk guidance,
and writes an **immutable, hash-chained evidence trail** — all surfaced through an
enterprise multi-page Streamlit console and a FastAPI backend.

The Live Conversation Monitor is a **real call**: continuous speech in, semantic
end-of-utterance detection, prosody-informed sentiment, live PII redaction, a
streamed reply, and a spoken answer in an Azure neural voice.

**The multilingual claim is narrow and deliberate.** The customer only ever hears
wording a compliance reviewer approved, in the language the call is being held
in. Arabic and Urdu are *authored* in the policy corpus beside the clause they
come from — not produced by a translation model at call time, because a sentence
generated on the call is a sentence nobody signed off. Where a language has no
approved wording, the agent falls back to English and records that it did: a gap
for a reviewer to close, rather than a fluent sentence nobody wrote.

It reasons on two independent axes — **vulnerability** and **financial
detriment** (arrears, essential-spend conflict, scam exposure, income shock) —
because a financially literate customer can be deep in arrears, and a bereaved
customer may have no financial stress at all.

> Every external integration is **optional**. Unset services fall back to safe
> local implementations, so `streamlit run app.py` works out of the box and each
> integration lights up (🟢) the moment you configure it. See the **Settings**
> page for live status of every component.

## Where the regulation sits

The bank is fictional (**Gulf Union Bank**) and all data is synthetic, but the
obligations are real ones. The **four vulnerability drivers** — health, life
events, resilience, capability — are the bank's *own* adopted framework, borrowed
from the UK regulator's guidance because it is the most fully worked-out
articulation of the idea anywhere. The **duties that follow** are grounded in UAE
instruments: the CBUAE Consumer Protection Regulation (Circular 8/2020) and the
Consumer Protection Standards issued under it, Circular 29/2011 on loans to
individual customers, the Mortgage Loan Regulations 31/2013, Al Etihad Credit
Bureau reporting, and Sanadak, the independent Ombudsman Unit.

That split is deliberate and stays visible throughout: **the taxonomy is the
bank's, the duties are the regulator's.** Where a rule is the bank's own policy
rather than a published regulation — the 09:00–20:00 calling window, for
instance — it is labelled as the approved treatment strategy, because a reviewer
asking "why 20:00?" should be pointed at the strategy document and not at a
regulation that does not say it.

**The borrower base shapes the product.** Most of these customers are expatriate
residents whose right to remain follows their employment. Job loss is an income
shock and a countdown at once; many repay through a salary transfer that stops
when the salary does; a remittance to family abroad is an essential cost, not
discretionary spending; and a collections call to an employer is not an
embarrassment but a threat to the customer's residency. None of that is in a UK
or Indian collections playbook, and all of it is in this one.

---

## Tech stack

| Layer | Technology | Fallback when unconfigured |
|-------|-----------|----------------------------|
| Frontend | **Streamlit** (multipage) | — |
| Backend API | **FastAPI** | — |
| LLM | **Claude** (`claude-opus-5`, adaptive thinking, streaming) **or OpenRouter** (any model), structured JSON | deterministic classifiers |
| Agent framework | **LangGraph** `StateGraph` | sequential runner |
| Embeddings | **Azure OpenAI** → **sentence-transformers** (local, semantic) | hashing embedder |
| Speech in | **Azure Speech** continuous recognition, per-language locale (browser SDK + short-lived token) | Web Speech API → push-to-talk |
| Speech out | **Azure Speech** neural TTS, voice chosen by call language, SSML styles | text only |
| Languages | English · Arabic · Urdu · Hindi · Malayalam · Filipino — script-first detection, no model | English |
| Prosody | standard-library DSP · Web Audio API | text channel |
| Database | **PostgreSQL** | local SQLite |
| Vector DB | **ChromaDB** (persistent) | in-memory cosine store |
| Observability | **Langfuse** | in-memory event buffer |
| Logging | **MLflow / OpenTelemetry** | Python logging |
| Guardrails | **NeMo Guardrails / custom validators** | custom validators (default) |

## The 12 pages

Executive Dashboard · **Live Conversation Monitor** · **Collections Queue** ·
Vulnerability Detection · AI Guidance Panel · Human Approval Queue · Policy
Knowledge Base (RAG search) · Guardrails Dashboard · Audit Trail · Customer
Timeline · Analytics & Evaluation · Settings.

The **Collections Queue** gives equal space to the calls the system *will not*
make. A Head of Collections signing this off does not need reassurance that it
dials; they need to see that at 21:00 it dials nothing, that a customer who asked
to be left alone has gone from the list, and that each of those is attributable
to a named gate rather than to a coincidence of timing. The clock on that page
moves, because calling-hours conduct is invisible at any single moment.

The Live Conversation Monitor carries three channels — **Call**, **Chat** and
**Library** — over one pipeline, with a live signal rail showing the journey and
its obligations, the four driver scores, the customer's measured distress, what
the voice is doing that the words are not, and what has been redacted out of the
transcript.

## Architecture

```
                         ┌──────────── Streamlit console (12 pages) ───────────┐
                         │  FastAPI  ──────────────── same pipeline ───────────┘
                         ▼
   utterance ──▶ ┌─────────────────── LangGraph pipeline ────────────────────┐
   (spoken       │ Conversation → Financial Context → Sentiment →             │
    or typed)    │ Vulnerability → Policy(RAG) → Guidance →                   │
                 │ Compliance → Supervisor → Evidence                         │
                 └───┬──────────┬───────────┬───────────┬──────────┬─────────┘
        input guardrails    journey +   ChromaDB      LLM      output guardrails
        (PII incl. spoken/  drivers     (journey-    (advice)  (confidence/
         injection/toxicity)            re-ranked)             grounding/
                                                               prohibited action/
                                                               approval)
                                     │
                                     ▼
                    Immutable hash-chained evidence  ──▶  PostgreSQL / SQLite
                    Audit log · Observability (Langfuse/MLflow)
```

**Agents** (`src/guardiancx/agents/`)

1. **Conversation** — ingests the turn, runs input guardrails, masks PII.
2. **Account Access** — what the caller asked the ledger for, whose money it is,
   and how much may be said. Refuses a third party's data outright.
3. **Financial Context** — product · banking journey · financial-stress
   indicators · arrears; maps the journey to the obligations it engages.
4. **Sentiment** — fuses what was said with how it was said (prosody) into
   valence, arousal and a calibrated distress score.
5. **Vulnerability Detection** — structured JSON or heuristic; the 4 drivers.
6. **Policy Retrieval** — RAG over ChromaDB, filtered by driver and re-ranked by
   journey; a high-harm journey retrieves even when no driver fired.
7. **Guidance** — drafts an adaptation grounded strictly in retrieved policy,
   told the journey's obligations and its prohibited actions.
8. **Compliance** — runs output guardrails on the recommendation.
9. **Supervisor** — sets risk and routes to human approval on any of five
   grounds: high risk, a guardrail block, low confidence, customer distress, or
   financial detriment already occurring.
10. **Evidence Logger** — writes the immutable, hash-chained record + audit event.

**Guardrails** (`src/guardiancx/guardrails/`): `pii`, `injection`, `toxicity`
(input); `confidence`, `hallucination`, `prohibited_action`, `approval` (output);
`credential_request`, `approved_wording`, `data_disclosure` and
`customer_clarity` (reply).

The reply set is ordered by what cannot be undone. `credential_request` runs
first and blocks any reply asking the customer for a PIN, password, OTP or CVV —
the approved challenge flow confirms identity from what the bank already knows,
and asking for a secret teaches the customer to answer the next impersonator who
does. `approved_wording` holds an outbound call to the wording its strategy
authorised, in the language the call is being held in; it accepts composition of
approved fragments and rejects invention. `data_disclosure` blocks any identifier
from the account book reaching the customer. `customer_clarity` warns about a
draft they cannot act on.

**Outbound collections** (`src/guardiancx/outbound/`): the channel that *starts*
the call. `strategy` holds the named, versioned treatment strategies — contact
limits, clause allow-lists, and the handover triggers no strategy may switch off;
`contact_rules` owns the approved calling window in Gulf Standard Time and when
it next opens; `optout` is the durable, append-only register plus the detector
that hears "stop calling me" mid-call in Arabic, Urdu or Hindi; `queue` is the
only route from a scheduled obligation to a dialled call, and it records the gate
that refused whenever a task does not pass.

The gate order is load-bearing: consent is checked before calling hours, because
recording "outside calling hours" for a customer who has withdrawn consent
implies the bank means to try again tomorrow.

**Consent** (`src/guardiancx/agents/consent.py`): every call opens by asking to
record, and nothing is assessed until the customer answers. A refusal ends the
call; a question is not consent.

**Customer ledger** (`src/guardiancx/finance/accounts.py`): synthetic customers,
accounts, balances, arrears and transactions — including a joint mortgage and a
sole card in a deceased spouse's name, which is what the third-party refusal is
tested against. **Language is a ledger field**, not a call-time guess: an outbound
call must choose a voice before the customer has said anything, and defaulting to
English is how a multilingual channel quietly becomes an English one with a
translation feature.

**Customer-facing reply** (`src/guardiancx/agents/reply.py`): policy is written
for handlers, so the reply is composed rather than quoted. Each clause carries
approved `Offer:` wording used verbatim; derivation fills the gap for clauses
without it. Acknowledge, at most two concrete offers, then hand the turn back.

**Finance domain** (`src/guardiancx/finance/`): products, 10 banking journeys, 10
financial-stress indicators, the journey → CBUAE obligation map, and the
prohibited actions each journey carries — including the two this market adds:
never contact the customer's employer or sponsor about a debt, and never tell a
customer their residency is at risk because of it.

**Live call** (`src/guardiancx/voice/`): `endpointing` (semantic EOU),
`prosody` (acoustic distress), `live_pii` (spoken + written redaction on the
streaming transcript — Emirates ID, UAE IBAN and mobile, dictated digit runs).

**Language** (`src/guardiancx/utils/language.py`): the six languages the book
speaks, their recogniser locales and neural voices, and script-first detection.
Arabic and Urdu share an alphabet and are separated by the handful of letters
Urdu added — without that, every Urdu speaker is answered in Arabic by a
confident system. Detection *proposes*; the customer record and a strong signal
decide, because switching a distressed caller into the wrong language on the
strength of one short utterance is its own kind of failure.

## Quickstart

```bash
python -m venv .venv && .venv\Scripts\Activate.ps1     # Windows PowerShell
python -m pip install -r requirements.txt

cp .env.example .env        # optional — fill in any keys you have

python scripts/seed_data.py --fresh   # index policy + run synthetic conversations
streamlit run app.py                  # open http://localhost:8501
```

Optional FastAPI backend:

```bash
uvicorn api:app --reload    # http://localhost:8000/docs
```

> Use `python -m pip` on this machine (bare `pip` points at a different Python).

## Configuration

All via environment variables / `.env` (see `.env.example`). Highlights:

- `ANTHROPIC_API_KEY` — enables Claude for detection + guidance, **or**
- `OPENROUTER_API_KEY` + `OPENROUTER_MODEL` — use any OpenRouter model instead
  (used only when the Anthropic key is empty). With neither, the app runs on the
  deterministic heuristic.
- `AZURE_OPENAI_*` — Azure OpenAI embeddings for RAG (otherwise a local
  sentence-transformers model is used; the hashing embedder is the last resort).
- `AZURE_SPEECH_KEY` / `AZURE_SPEECH_REGION` — continuous speech recognition
  and the spoken reply. Without them the call falls back to the browser's own
  recogniser (Chrome/Edge) and the reply is text only.
- `GUARDIANCX_DATABASE_URL` — Postgres URL (else SQLite file).
- `GUARDIANCX_CONFIDENCE_THRESHOLD`, `GUARDIANCX_HIGH_RISK_APPROVAL` — guardrail tuning.

Agent system prompts live in `src/guardiancx/agents/prompts.py`. The product
roadmap is in [ROADMAP.md](ROADMAP.md).

## Deploying to Streamlit Community Cloud

Configuration on the cloud comes from **Streamlit secrets**, not a `.env` file.
In the app's **Settings → Secrets**, paste the keys you need (top-level, named
exactly as in [.streamlit/secrets.toml.example](.streamlit/secrets.toml.example)):

```toml
OPENROUTER_API_KEY = "sk-or-..."
OPENROUTER_MODEL   = "anthropic/claude-sonnet-4.5"
ANTHROPIC_API_KEY  = "sk-ant-..."   # preferred; OpenRouter is the alternative
AZURE_SPEECH_KEY   = "..."
AZURE_SPEECH_REGION = "eastus"
```

The app reads `st.secrets` and hydrates them into its settings automatically, so
the same code path works locally (`.env`) and on the cloud (secrets). Main file:
`app.py`.

Notes for the cloud:
- **The call runs in the browser.** Continuous recognition uses the Azure Speech
  JS SDK with a short-lived token minted server-side (the subscription key never
  reaches the page); push-to-talk uses `st.audio_input` → Azure Speech REST.
  Neither needs a microphone on the host.
- Embeddings fall back to the local hashing embedder unless you add
  `sentence-transformers` (heavier) or Azure OpenAI embeddings; for best
  retrieval quality run locally or configure Azure OpenAI.

## Extending

- **Add a guardrail** — subclass `guardrails.base.Guardrail`, implement `check`,
  and register it in `guardrails/manager.py` (`default_input_guardrails` /
  `default_output_guardrails`). Nothing else changes.
- **Add an agent** — write a `run(state) -> state` node in `agents/`, then add it
  to `PIPELINE` in `agents/graph.py`; it slots into the LangGraph automatically.
- **Add policy** — drop a markdown file in `data/policies/` and click *Re-ingest
  policies* on the Settings page (or run the seed script).

## Compliance posture

- **Advisory only** — the system recommends; a human decides. It never acts on
  the account. The outbound channel schedules and places calls; it does not move
  money, vary an agreement, or record a decision the customer has not been given
  in approved wording.
- **Approved wording only** — on a strategy that requires it, every commitment
  the customer hears is checked against the signed-off wording for their
  language, and an invented offer is blocked rather than logged.
- **Every opt-out honoured** — the register is durable and append-only, written
  the moment the customer says so rather than at wrap-up, and consulted per task
  rather than per batch so a request made on one line stops a call queued on
  another.
- **Permitted hours enforced, not documented** — the calling window is a gate in
  front of the dialler. A refusal names the rule and the time it next opens.
- **Grounded** — guidance must cite retrieved policy; ungrounded citations are
  blocked by the hallucination guardrail.
- **Human-in-the-loop** — high-risk guidance is gated behind explicit approval.
- **Tamper-evident** — evidence is append-only and hash-chained; the Audit Trail
  verifies the chain live.
- **All data is synthetic.**

## Tests

```bash
python -m pytest -q          # 278 tests: guardrails, pipeline, finance, voice, reply,
                             # multilingual, outbound conduct, UI
python scripts/eval_rag.py   # retrieval quality, with and without journey re-ranking
```

Retrieval over the rewritten CBUAE corpus: **hit-rate@3 = 1.000**, MRR 0.926
across 36 labelled queries, with journey-aware re-ranking worth +0.051 MRR.

`tests/test_live_monitor.py` drives the live console headlessly with Streamlit's
`AppTest`, so the page a reviewer will spend their time on is covered end to end —
a chat turn through all nine agents, PII never reaching the transcript, and a
session saved into the conversation library.
