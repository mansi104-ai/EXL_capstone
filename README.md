# 🛡️ GuardianCX

**Vulnerable-customer protection for UK retail banking and consumer credit.**

GuardianCX detects and assists **vulnerable customers** *during* the call. A
nine-agent LangGraph pipeline classifies the banking situation (product, journey,
financial-stress indicators) and the customer's vulnerability across the four FCA
drivers, retrieves the firm's prescribed adaptation via journey-aware RAG,
enforces a seven-guardrail stack, requires human approval for high-risk guidance,
and writes an **immutable, hash-chained evidence trail** — all surfaced through an
enterprise multi-page Streamlit console and a FastAPI backend.

The Live Conversation Monitor is a **real call**: continuous speech in, semantic
end-of-utterance detection, prosody-informed sentiment, live PII redaction, a
streamed reply, and a spoken answer in an Azure neural voice.

It reasons on two independent axes — **vulnerability** (FG21/1) and **financial
detriment** (arrears, essential-spend conflict, scam exposure, gambling harm) —
because a financially literate customer can be deep in arrears, and a bereaved
customer may have no financial stress at all.

> Every external integration is **optional**. Unset services fall back to safe
> local implementations, so `streamlit run app.py` works out of the box and each
> integration lights up (🟢) the moment you configure it. See the **Settings**
> page for live status of every component.

---

## Tech stack

| Layer | Technology | Fallback when unconfigured |
|-------|-----------|----------------------------|
| Frontend | **Streamlit** (multipage) | — |
| Backend API | **FastAPI** | — |
| LLM | **Claude** (`claude-opus-5`, adaptive thinking, streaming) **or OpenRouter** (any model), structured JSON | deterministic classifiers |
| Agent framework | **LangGraph** `StateGraph` | sequential runner |
| Embeddings | **Azure OpenAI** → **sentence-transformers** (local, semantic) | hashing embedder |
| Speech in | **Azure Speech** continuous recognition (browser SDK + short-lived token) | Web Speech API → push-to-talk |
| Speech out | **Azure Speech** neural TTS with SSML speaking styles | text only |
| Prosody | standard-library DSP · Web Audio API | text channel |
| Database | **PostgreSQL** | local SQLite |
| Vector DB | **ChromaDB** (persistent) | in-memory cosine store |
| Observability | **Langfuse** | in-memory event buffer |
| Logging | **MLflow / OpenTelemetry** | Python logging |
| Guardrails | **NeMo Guardrails / custom validators** | custom validators (default) |

## The 11 pages

Executive Dashboard · **Live Conversation Monitor** · Vulnerability Detection ·
AI Guidance Panel · Human Approval Queue · Policy Knowledge Base (RAG search) ·
Guardrails Dashboard · Audit Trail · Customer Timeline · Analytics & Evaluation ·
Settings.

The Live Conversation Monitor carries three channels — **Call**, **Chat** and
**Library** — over one pipeline, with a live signal rail showing the journey and
its obligations, the four driver scores, the customer's measured distress, what
the voice is doing that the words are not, and what has been redacted out of the
transcript.

## Architecture

```
                         ┌──────────── Streamlit console (11 pages) ───────────┐
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
2. **Financial Context** — product · banking journey · financial-stress
   indicators · arrears; maps the journey to the obligations it engages.
3. **Sentiment** — fuses what was said with how it was said (prosody) into
   valence, arousal and a calibrated distress score.
4. **Vulnerability Detection** — structured JSON or heuristic; 4 FCA drivers.
5. **Policy Retrieval** — RAG over ChromaDB, filtered by driver and re-ranked by
   journey; a high-harm journey retrieves even when no driver fired.
6. **Guidance** — drafts an adaptation grounded strictly in retrieved policy,
   told the journey's obligations and its prohibited actions.
7. **Compliance** — runs output guardrails on the recommendation.
8. **Supervisor** — sets risk and routes to human approval on any of five
   grounds: high risk, a guardrail block, low confidence, customer distress, or
   financial detriment already occurring.
9. **Evidence Logger** — writes the immutable, hash-chained record + audit event.

**Guardrails** (`src/guardiancx/guardrails/`): `pii`, `injection`, `toxicity`
(input); `confidence`, `hallucination`, `prohibited_action`, `approval` (output).

**Finance domain** (`src/guardiancx/finance/`): products, 10 banking journeys, 10
financial-stress indicators, the journey → regulation map (CONC 7.3, Consumer
Duty PRIN 2A, Breathing Space, APP-fraud reimbursement, BCOBS, MCOB), and the
prohibited actions each journey carries.

**Live call** (`src/guardiancx/voice/`): `endpointing` (semantic EOU),
`prosody` (acoustic distress), `live_pii` (spoken + written redaction on the
streaming transcript).

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

- **Advisory only** — the system recommends; a human decides. It never acts.
- **Grounded** — guidance must cite retrieved policy; ungrounded citations are
  blocked by the hallucination guardrail.
- **Human-in-the-loop** — high-risk guidance is gated behind explicit approval.
- **Tamper-evident** — evidence is append-only and hash-chained; the Audit Trail
  verifies the chain live.
- **All data is synthetic.**

## Tests

```bash
python -m pytest -q          # 76 tests: guardrails, pipeline, finance, voice, UI
python scripts/eval_rag.py   # retrieval quality, with and without journey re-ranking
```

`tests/test_live_monitor.py` drives the live console headlessly with Streamlit's
`AppTest`, so the page a reviewer will spend their time on is covered end to end —
a chat turn through all nine agents, PII never reaching the transcript, and a
session saved into the conversation library.
