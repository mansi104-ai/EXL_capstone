# GuardianCX — How it was built

> A multi-agent advisory system for **UK retail banking and consumer credit**. It
> detects vulnerable customers *during* the call, grounds its guidance in the
> firm's policy and the rules that apply to that journey, keeps a human in the
> loop, and evidences that the customer was treated fairly.

## The problem, narrowed

A general-purpose classifier can tell you that a customer sounds distressed. It
cannot tell you that a customer three payments behind on a regulated credit
agreement, choosing between the mortgage and the electricity bill, has engaged
CONC 7.3's forbearance duty — and that offering them a consolidation loan would
breach the Consumer Duty.

That gap is the product. GuardianCX reasons on **two axes at once**:

| Axis | What it measures | Source |
|---|---|---|
| **Vulnerability** | health · life events · resilience · capability | FCA FG21/1 |
| **Financial detriment** | arrears · essential-spend conflict · income shock · scam exposure · gambling harm | the firm's own operational data |

They are independent, and the intersection is where a firm must act. A composed,
financially literate customer can be in serious arrears. A bereaved customer may
have no financial stress at all. Systems that collapse the two axes mis-serve
both.

## Architecture — a turn's journey

```mermaid
flowchart TD
    IN["Customer turn<br/>speak · type · upload"] --> RED["Live PII redaction<br/><i>spoken + written forms, on the way in</i>"]
    RED --> A1
    subgraph P["LangGraph pipeline · 9 agents"]
      direction TB
      A1["1 · Conversation<br/><i>masking · input guardrails</i>"] --> A2["2 · Financial Context<br/><i>product · journey · stress indicators</i>"]
      A2 --> A3["3 · Sentiment<br/><i>words + voice → distress</i>"]
      A3 --> A4["4 · Vulnerability Detection<br/><i>4 FCA drivers</i>"]
      A4 -->|drivers + journey| A5["5 · Policy Retrieval — RAG<br/><i>driver-filtered, journey-re-ranked</i>"]
      A5 -->|policy clauses| A6["6 · Guidance<br/><i>grounded only in retrieved policy</i>"]
      A6 --> A7["7 · Compliance<br/><i>confidence · grounding · prohibited actions</i>"]
      A7 --> A8["8 · Supervisor<br/><i>five routes to human approval</i>"]
      A8 --> A9["9 · Evidence Logger<br/><i>append-only, hash-chained</i>"]
    end
    A9 --> OUT["Streamed reply → Azure neural voice"]
    A9 --> EV["Evidence store → 11 dashboards · audit · approvals"]
```

Only customer turns are assessed. A parallel **FastAPI** service exposes the same
pipeline. **Every stage degrades gracefully:** no LLM key → deterministic
classifiers; no ChromaDB → in-memory store; no embeddings model → hashing; no
Azure Speech → the browser's own recogniser; no Postgres → SQLite. The app runs
end-to-end with zero credentials, and each integration activates when configured.

## The live call

The Live Conversation Monitor is a real call, not a form. Speech in → nine
agents → speech out, per turn.

| Piece | What it does | Why it is not the obvious thing |
|---|---|---|
| **Continuous capture** | Azure Speech streams interim words as they are spoken | Push-to-talk cannot show the customer being interrupted, or PII being caught mid-sentence |
| **EOU model** | decides the turn is over from *meaning*, not silence | Silence endpointing talks over a bereaved customer pausing after "my husband passed away and…". Thresholds adapt to how finished the sentence looks, and extend further when the caller is distressed |
| **Prosody** | six acoustic measurements → agitation, tremor, hesitancy | The transcript flattens the call. A shaking voice weighs more than a loud one — an angry customer is not a vulnerable one |
| **Live PII** | spoken *and* written forms redacted on the way in | "my sort code is oh nine, oh one, double two" matches no written pattern. The guard also fires on the *announcement*, before the value is spoken |
| **Streaming** | the reply renders as the model writes it | The synthesiser can start on the first sentence instead of the last |
| **Spoken reply** | Azure neural voice, style chosen by the customer's state | Recognition is muted during playback, or the agent transcribes and answers itself |

The endpointing model and the prosody measurements run in the browser *and* in
Python: the browser copy makes the millisecond timing call, Python re-runs the
authoritative model for the trace and the evidence record. Thresholds live in
Python only, so the two cannot drift.

## Technology

| Layer | Technology | Fallback |
|-------|-----------|----------|
| Interface | **Streamlit** (11 pages) · Plotly · a custom voice component | — |
| API | **FastAPI** | — |
| LLM | **Claude** (`claude-opus-5`, adaptive thinking, per-call-site effort) **or OpenRouter** · structured JSON · streaming | deterministic classifiers |
| Orchestration | **LangGraph** `StateGraph` (9 agents) | sequential runner |
| Vector DB | **ChromaDB** (cosine) + journey re-ranking | in-memory store |
| Embeddings | **sentence-transformers** / **Azure OpenAI** | hashing embedder |
| Speech in | **Azure Speech** continuous recognition (browser SDK, short-lived token) | Web Speech API · push-to-talk REST |
| Speech out | **Azure Speech** neural TTS with SSML style | text only |
| Prosody | standard-library DSP · Web Audio API | text channel |
| Database | **SQLAlchemy** — Postgres / SQLite, additive migrations | SQLite |
| Evidence | append-only, **hash-chained** log | — |
| Guardrails | PII · injection · toxicity · confidence · grounding · **prohibited actions** · human approval | custom validators (NeMo optional) |
| Observability | **Langfuse** · **MLflow / OpenTelemetry** | in-memory buffer |
| Config | **pydantic-settings** | `.env` / `st.secrets` |

No new third-party dependency was added for any of the live-call work.

## Guardrails

| Stage | Guardrail | Severity | Purpose |
|-------|-----------|----------|---------|
| Input | PII masking | warn | Redact emails, cards, sort codes, NI numbers, postcodes, dates of birth, IBANs — spoken forms included |
| Input | Prompt injection | block | Detect instruction-override / prompt-extraction |
| Input | Toxicity | warn | Flag abusive language for tone-aware handling |
| Output | Confidence threshold | warn | Route low-confidence guidance to review |
| Output | Hallucination grounding | block | Reject citations not present in retrieved policy |
| Output | **Prohibited action** | block | Block advice that causes harm however well grounded — credit to a customer disclosing gambling harm, a "safe account" instruction on a scam call, a demand for a payment that would leave essentials unpaid |
| Output | Human approval | block | Mandatory sign-off for high-risk recommendations |

The prohibited-action check is deterministic and journey-scoped. It excludes
verbatim policy quotations and negated mentions, so a clause that *forbids* an
action is not mistaken for one proposing it — the prompt asks, the guardrail
guarantees.

## Retrieval quality — measured, not asserted

A labelled test set (query → acceptable policy clauses) is scored by
[`scripts/eval_rag.py`](../scripts/eval_rag.py) and on the Analytics page. The
harness runs the same set **twice**, so the contribution of the finance layer is
visible rather than claimed:

| Configuration | hit-rate@3 | precision@3 | recall@3 | MRR |
|---|---|---|---|---|
| Driver only — what a general tool can do | 1.000 | 0.927 | 0.927 | 0.922 |
| **Journey-aware — production** | **1.000** | **0.953** | **0.953** | **0.953** |

Both find a correct clause; journey-aware retrieval ranks it higher. Labels are
*sets*, because the corpus genuinely contains more than one correct answer to
some queries — scoring the second-best correct clause as a miss would measure the
label, not the retrieval.

## By the numbers

| | |
|---|---|
| LangGraph agents | **9** |
| Guardrails | **7** |
| Console pages | **11** |
| Policy clauses | **34** |
| FCA drivers · banking journeys · stress indicators | **4 · 10 · 10** |
| RAG hit-rate@3 | **100%** |
| Retrieval MRR | **0.95** |
| Tests passing | **76** |

## How it was built

1. **Evidence-first MVP** — streaming detection → policy lookup → advisory
   guidance → an immutable, hash-chained evidence trail.
2. **GuardianCX — modular agentic rebuild** — a clean package
   (`agents · guardrails · rag · services · database · ui`), a LangGraph
   pipeline, ChromaDB RAG, the guardrail stack, an 11-page console + FastAPI.
3. **Live & multi-provider** — real-time chat, browser speech capture, and a
   pluggable LLM, all with fallbacks.
4. **Deployment alignment** — one clean repo, configuration wired to Streamlit
   secrets so the same code runs locally and in the cloud.
5. **Retrieval fix & evaluation** — diagnosed poor retrieval (a hashing
   fallback), switched to semantic embeddings, added the evaluation harness.
6. **The finance niche & the real call** — narrowed to UK retail banking and
   consumer credit with a product/journey/detriment taxonomy, a Financial Context
   agent and journey-aware retrieval; and turned the monitor into a genuine
   call — continuous speech in, semantic endpointing, prosody-informed sentiment,
   live PII redaction, a streamed reply spoken back in a neural voice, and a
   conversation library that keeps what was said.

---

*GuardianCX is an internal decision-support tool: advisory only — a human handler
makes every customer-facing decision. All data shown is synthetic.*
