# GuardianCX — Product Roadmap

Prioritised improvements, most impactful first. Each item notes the problem it
solves and roughly where it lands in the codebase.

## P0 — Correctness & trust (do next)

1. **RAG evaluation harness.** A labelled set of utterances → expected policy
   clause(s), with precision@k / MRR reported in the Analytics page and CI. This
   is how we *prove* retrieval quality (the bereavement→VP-L1 class of bug) rather
   than spot-checking. *(new `eval/` + `scripts/eval_rag.py`)*
2. **Detection evaluation + threshold tuning.** Labelled driver data, per-driver
   precision/recall, and a calibration curve so `confidence`/thresholds are set
   from data, not guesses. *(`scripts/eval_detection.py`)*
3. **Grounding hardening.** Sentence-level citation checking (does each adaptation
   sentence trace to a cited clause?), not just "was the reference retrieved".
   Strengthens the hallucination guardrail. *(`guardrails/hallucination.py`)*
4. **LLM reliability.** Timeouts, retries with backoff, and a circuit-breaker on
   the LLM client; cache identical detections; show provider latency/health on
   Settings. *(`services/claude_client.py`)*

## P1 — Product depth

5. **Conversation-level context.** Assess the running conversation, not just the
   latest turn (a signal often spans turns); de-duplicate repeat detections;
   maintain a per-conversation vulnerability profile. *(`agents/` + new state)*
6. **Full audit export & case view.** Per-conversation case file (transcript +
   every decision + approvals) exportable as PDF for supervisory review.
   *(`reporting/`, `ui/views.py`)*
7. **Approval workflow.** Roles (handler vs supervisor), SLA timers on the queue,
   reason codes, and notifications; write approver identity to evidence. *(`database/`)*
8. **Policy management UI.** Upload/enable/disable policy documents and see the
   re-index result in-app; versioned policy sets. *(`rag/`, `ui/views.py`)*
9. **Real-time streaming speech.** Continuous Azure transcription (not just
   single-phrase), with live partial transcripts in the chat. *(`services/speech.py`)*

## P2 — Enterprise hardening

10. **AuthN/AuthZ & multi-tenant.** SSO, per-workspace data isolation, row-level
    security on evidence.
11. **Observability end-to-end.** Wire Langfuse traces + MLflow/OpenTelemetry
    metrics through every agent span; dashboards for drift and cost.
12. **PII posture.** Move masking before any network egress at the transport
    layer; configurable entity set; reversible tokenisation with a vault.
13. **Guardrail depth.** Swap the toxicity lexicon for a hosted classifier; add a
    jailbreak/secret-exfiltration guardrail; NeMo Guardrails rails around the LLM.
14. **Deployment.** Dockerfile + compose (app, Postgres, Chroma), health checks,
    and a CI pipeline running the eval harnesses on every PR.

## P3 — Model & UX polish

15. **Model picker in Settings** (switch OpenRouter models / providers live).
16. **Bias & fairness testing** across demographic-style paraphrases of the same
    disclosure; report disparity.
17. **Feedback loop.** Let handlers rate guidance; use ratings to tune prompts and
    thresholds (human-in-the-loop learning).
18. **Accessibility & localisation** of the console itself (WCAG, multi-language).

---

### How to work this roadmap
- P0 items are about *proving and protecting correctness* — they should gate any
  "it works" claim. Start with the RAG eval harness (#1).
- Each item is intentionally scoped to one area so it can be a single PR with its
  own tests. Add a guardrail or agent by following the "Extending" section of the
  README — the architecture is designed for it.
