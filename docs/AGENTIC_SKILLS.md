# Agentic CX Skills — Capstone Showcase

This MVP is the capstone for the Agentic CX training. It is deliberately built to
demonstrate the core competencies from that programme in a single, coherent,
regulated-CX product. This document maps each skill to where it lives in the
code and how to see it in the running app.

| # | Agentic CX skill | Where it's demonstrated | See it in the app |
|---|------------------|-------------------------|-------------------|
| 1 | **Agent loop & orchestration** | [`src/vca/agent.py`](../src/vca/agent.py) — `CareAgent.run_turn` runs a 6-step loop: perceive → assess → retrieve → advise → decide → record | Expand any customer turn's **agent trace** |
| 2 | **Tool use** | The classifier and the RAG retriever are the agent's tools, invoked per turn ([`pipeline.py`](../src/vca/pipeline.py) `assess`) | ASSESS + RETRIEVE steps in the trace |
| 3 | **Retrieval-Augmented Generation (RAG)** | [`src/vca/rag/`](../src/vca/rag/) — policy parsed into driver-scoped clauses, retrieved semantically (MiniLM + FAISS) with a TF-IDF fallback | RETRIEVE step shows the policy clause + score |
| 4 | **Grounding & citations** | Every adaptation carries its `policy_reference` (e.g. `VP-L1`) back to the firm's policy — no ungrounded advice | Guidance card shows `[VP-…]` references |
| 5 | **Guardrails / safety** | Advisory-only by construction (`Guidance.advisory_only`); the agent never acts. Human-in-the-loop decision required before anything is recorded | "guidance only, you decide" + Accept/Modify/Dismiss |
| 6 | **Human-in-the-loop** | Handler records Accept / Modify / Dismiss; only then is evidence written ([`record_outcome`](../src/vca/pipeline.py)) | Decision buttons on each flagged turn |
| 7 | **Observability / transparency** | Full per-turn trace with per-driver confidences and retrieval scores ([`agent.py`](../src/vca/agent.py) `TurnTrace`) | Driver score bars + trace steps |
| 8 | **Evaluation** | Per-driver precision/recall/F1 harness ([`scripts/evaluate.py`](../scripts/evaluate.py)); training reports validation metrics | `python scripts/evaluate.py` |
| 9 | **Fine-tuning a classifier** | DistilBERT multi-label fine-tune ([`src/vca/classifier/train.py`](../src/vca/classifier/train.py)) with a keyword fallback | Sidebar shows 🟢 Transformer once trained |
| 10 | **Multi-modal / multi-channel input** | Type, upload (JSON/CSV/TXT), replay, or **live Azure Speech** ([`ingestion/`](../src/vca/ingestion/)) | "Converse" tab input modes |
| 11 | **Real-time streaming** | Azure continuous recognition streamed into the pipeline ([`azure_speech.py`](../src/vca/ingestion/azure_speech.py) `LiveAzureRecognizer`) | "Live microphone" mode |
| 12 | **Auditability & trust** | Append-only, **hash-chained** evidence log with tamper verification ([`evidence/store.py`](../src/vca/evidence/store.py)) | Sidebar chain status + Evidence table |
| 13 | **Aggregate insight / reporting** | Portfolio-level fair-treatment metrics ([`reporting/metrics.py`](../src/vca/reporting/metrics.py)) | "Evidence & report" tab |
| 14 | **Graceful degradation** | Every heavy dependency (transformer, embeddings, Azure) has a working fallback, so the agent always runs | Sidebar engine status (🟢 live / 🟡 fallback) |
| 15 | **Deployability** | Streamlit Cloud-ready with a slim requirement set that runs on fallbacks | `README.md` deploy section |

## The agent loop, concretely

For each **customer** utterance:

```
👂 PERCEIVE   ingest turn (typed / uploaded / spoken via Azure)
🧭 ASSESS     classifier tool → confidence per driver (health/life/resilience/capability)
📚 RETRIEVE   RAG tool → firm policy clause(s) for the triggered driver(s)
💬 ADVISE     compose discreet, advisory-only guidance (with policy citation)
🧑 DECIDE     handler accepts / modifies / dismisses  ← human in the loop
🔏 RECORD     append detection + guidance + outcome to hash-chained evidence
```

Handler turns pass straight through — only the customer is assessed.

## Why this shape fits regulated CX

The regulator wants **evidenced fair treatment**, not automation of sensitive
decisions. So the agent is intentionally *advisory*: it perceives, reasons
transparently, grounds its advice in policy, and hands the decision to a human —
while making the whole thing auditable. That is the agentic-CX pattern applied
where trust and accountability matter most.
