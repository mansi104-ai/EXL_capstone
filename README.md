# 🛡️ GuardianCX

**Agentic Customer Experience for regulated financial-services conversations.**

GuardianCX detects and assists **vulnerable customers** in banking/insurance
service conversations. A LangGraph multi-agent pipeline classifies vulnerability
across the four FCA drivers, retrieves the firm's prescribed adaptation via RAG,
enforces a full guardrail stack, requires human approval for high-risk guidance,
and writes an **immutable, hash-chained evidence trail** — all surfaced through
an enterprise multi-page Streamlit console and a FastAPI backend.

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
| LLM | **Claude** (`claude-opus-4-8`, structured JSON) | keyword heuristic |
| Agent framework | **LangGraph** `StateGraph` | sequential runner |
| Embeddings | **Azure OpenAI** | deterministic hashing embedder |
| Speech | **Azure Speech** | text input |
| Database | **PostgreSQL** | local SQLite |
| Vector DB | **ChromaDB** (persistent) | in-memory cosine store |
| Observability | **Langfuse** | in-memory event buffer |
| Logging | **MLflow / OpenTelemetry** | Python logging |
| Guardrails | **NeMo Guardrails / custom validators** | custom validators (default) |

## The 11 pages

Executive Dashboard · Live Conversation Monitor · Vulnerability Detection ·
AI Guidance Panel · Human Approval Queue · Policy Knowledge Base (RAG search) ·
Guardrails Dashboard · Audit Trail · Customer Timeline · Analytics & Evaluation ·
Settings.

## Architecture

```
                         ┌──────────── Streamlit console (11 pages) ───────────┐
                         │  FastAPI  ──────────────── same pipeline ───────────┘
                         ▼
   utterance ──▶ ┌─────────────────── LangGraph pipeline ────────────────────┐
                 │ Conversation → Vulnerability → Policy(RAG) → Guidance →    │
                 │ Compliance → Supervisor → Evidence                         │
                 └───┬──────────┬───────────┬───────────┬──────────┬─────────┘
        input guardrails    Claude      ChromaDB     Claude    output guardrails
        (PII/injection/     (drivers)   (policy)     (advice)  (confidence/
         toxicity)                                              hallucination/
                                                                approval)
                                     │
                                     ▼
                    Immutable hash-chained evidence  ──▶  PostgreSQL / SQLite
                    Audit log · Observability (Langfuse/MLflow)
```

**Agents** (`src/guardiancx/agents/`)

1. **Conversation** — ingests the turn, runs input guardrails, masks PII.
2. **Vulnerability Detection** — Claude (structured JSON) or heuristic; 4 drivers.
3. **Policy Retrieval** — RAG over ChromaDB, scoped per triggered driver.
4. **Guidance** — Claude drafts adaptation grounded strictly in retrieved policy.
5. **Compliance** — runs output guardrails on the recommendation.
6. **Supervisor** — sets risk, routes high-risk/blocked cases to human approval.
7. **Evidence Logger** — writes the immutable, hash-chained record + audit event.

**Guardrails** (`src/guardiancx/guardrails/`): `pii`, `injection`, `toxicity`
(input); `confidence`, `hallucination`, `approval` (output).

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

- `ANTHROPIC_API_KEY` — enables Claude for detection + guidance.
- `AZURE_OPENAI_*` — enables Azure OpenAI embeddings for RAG.
- `GUARDIANCX_DATABASE_URL` — Postgres URL (else SQLite file).
- `GUARDIANCX_CONFIDENCE_THRESHOLD`, `GUARDIANCX_HIGH_RISK_APPROVAL` — guardrail tuning.

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
python -m pytest -q      # guardrails + end-to-end pipeline
```
