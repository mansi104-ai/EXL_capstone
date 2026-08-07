# Design & Compliance Notes — Vulnerable Customer Care Agent

## 1. Problem framing

Regulators (e.g. the FCA's *Guidance for firms on the fair treatment of
vulnerable customers*, FG21/1) require firms to identify customers in vulnerable
circumstances and adapt service accordingly. In practice, detection today
depends on an individual agent noticing a cue and remembering to flag it. This
is inconsistent and largely **unevidenced**: firms cannot demonstrate at
portfolio level that vulnerable customers received appropriate outcomes, which is
a live supervisory concern.

The VCA converts this judgement-dependent process into a **signal-based,
evidenced** one — without ever removing the human from the decision.

## 2. The four regulatory drivers

Detection is organised around the four regulator-defined drivers of
vulnerability:

| Driver | Examples of indicators |
|--------|------------------------|
| **Health** | serious/terminal illness, mental health, cognitive difficulty |
| **Life events** | bereavement, job loss, divorce, caring responsibilities |
| **Resilience** | financial distress, over-indebtedness, no savings buffer |
| **Capability** | low financial/digital literacy, language barrier, third-party help |

The classifier is **multi-label**: a single utterance can trip several drivers
(vulnerabilities frequently co-occur, e.g. bereavement + financial distress).

## 3. Architecture

```
audio / chat ──▶ Ingestion ──▶ Classifier ──▶ Guidance (RAG) ──▶ Handler
                (Azure/sim)     (4 drivers)     (policy lookup)      │
                                                                     ▼
                                              Evidence store ◀── Outcome
                                              (append-only,          │
                                               hash-chained)         ▼
                                                          Portfolio reporting
```

- **Ingestion** (`vca/ingestion`) — `AzureSpeechSource` for real-time
  transcription, `SimulatedSource` replaying transcripts/chat logs as fallback.
  A factory chooses based on config/credentials. Both yield `Utterance` objects.
- **Classifier** (`vca/classifier`) — fine-tuned DistilBERT multi-label model
  (`TransformerClassifier`) with a dependency-free `KeywordClassifier` fallback.
  A factory prefers the trained model when present. Detection is **advisory
  only** — it never triggers an action.
- **RAG** (`vca/rag`) — parses the firm's vulnerability policy into clauses
  tagged by driver, then retrieves the prescribed adaptation for each detected
  driver (`EmbeddingRetriever` via sentence-transformers/FAISS, or a TF-IDF
  `KeywordRetriever` fallback). Retrieval is **scoped to the detected driver** so
  guidance is on-point.
- **Guidance** (`vca/guidance`) — composes the discreet handler prompt from the
  detection + retrieved adaptations. Always marked `advisory_only=True`.
- **Evidence** (`vca/evidence`) — append-only, **hash-chained** JSONL log. Each
  record embeds the previous record's SHA-256 hash, forming a tamper-evident
  chain.
- **Reporting** (`vca/reporting`) — portfolio-level fair-treatment metrics over
  the evidence log, plus chain verification.
- **Pipeline** (`vca/pipeline.py`) — wires the stages. `assess()` does advisory
  classification + retrieval without writing evidence; `record_outcome()` commits
  detection + guidance + the handler's decision. Only customer utterances are
  assessed.
- **Agent** (`vca/agent.py`) — `CareAgent` orchestrates each customer turn as a
  transparent 6-step loop (perceive → assess → retrieve → advise → decide →
  record) and emits a `TurnTrace` the UI renders. See
  [AGENTIC_SKILLS.md](AGENTIC_SKILLS.md) for the full skills mapping.

## 4. Compliance-critical design decisions

### Advisory only, never automated action
The system surfaces guidance and stops. It does **not** suspend collections,
open a vulnerability case, apply a flag, or take any customer-facing action. The
`Guidance` object carries `advisory_only=True`; the handler decides and acts.
This keeps a human accountable for every outcome and avoids the regulatory and
fairness risks of automated adverse/benevolent action on an inferred signal.

### Tamper-evident evidence
The evidence log is append-only and hash-chained (`prev_hash` → `record_hash`).
`verify_chain()` recomputes every hash and detects any edit or reordering. This
is what turns "we think we treated people fairly" into "here is the immutable
record of every signal, the adaptation offered, what the handler did, and the
outcome." (`test_evidence.py` proves tampering breaks verification.)

### Evidenced fair treatment at portfolio level
`build_report()` aggregates detections by driver, handler outcomes, and
acceptance rate across conversations — the supervisory view of an otherwise
invisible population.

### Signal, not diagnosis
Scores are indicators, not clinical/financial determinations. Thresholds are
configurable (`config.yaml`), and the guidance text is explicit that it is a
possible signal for the handler to weigh, not a conclusion.

## 5. Data & privacy

All data in this repository is **synthetic** (`scripts/generate_data.py`). In a
real deployment: transcripts and vulnerability flags are special-category /
sensitive data; the evidence store would sit behind access controls and
retention policy, and the policy corpus would be the firm's own controlled
document.

## 6. Evaluation

`scripts/evaluate.py` reports per-driver precision/recall/F1 on the held-out
validation set for whichever classifier is active, so the fine-tuned transformer
can be compared against the keyword baseline.

## 7. Extending

- Swap the synthetic policy for the firm's real vulnerability policy — the RAG
  layer re-indexes automatically.
- Add diarization metadata in ingestion to attribute speakers precisely.
- Add a calibration step to set per-driver thresholds from labelled outcomes.
- Add reviewer sign-off fields to `Outcome` for a full audit workflow.
