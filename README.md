# Vulnerable Customer Care Agent (VCA)

An **advisory-only** agent that monitors regulated banking/insurance service
conversations for **vulnerability indicators** and helps handlers deliver — and
the firm evidence — consistent, fair treatment of vulnerable customers.

> Regulators require firms to identify customers in vulnerable circumstances
> (bereavement, serious illness, financial distress, cognitive difficulty) and
> adapt service accordingly. Today this depends on individual agents noticing a
> cue and remembering to flag it — inconsistent and largely unevidenced. VCA
> makes detection signal-based, surfaces the firm's prescribed adaptation, and
> writes an immutable evidence trail so fair treatment is demonstrable at
> portfolio level.

## What it does

1. **Ingests** a live conversation (real Azure Speech transcription, or a
   simulated transcript/chat feed as fallback).
2. **Detects** vulnerability indicators across the four regulator-defined
   drivers — **health, life events, resilience, capability** — with a
   fine-tuned transformer classifier. Detection is *signal-based and always
   advisory; it never takes automated action.*
3. **Retrieves** the firm's prescribed **adaptation** for each detected
   indicator via RAG over the internal vulnerability policy (e.g. *slow the
   pace, offer a trusted third party, suspend collections activity*).
4. **Surfaces** a discreet, advisory prompt to the handler.
5. **Records** every detection, the adaptation offered, the handler action and
   the outcome to an **append-only, hash-chained evidence log**.
6. **Reports** at portfolio level so the firm can demonstrate consistent
   treatment of an otherwise-invisible supervisory population.

## Architecture

```
                 ┌──────────────────────────────────────────────┐
                 │            Streamlit / FastAPI UI            │
                 │  live transcript · advisory prompts · report │
                 └───────────────▲───────────────▲──────────────┘
                                 │               │
   audio / text                 │ guidance      │ metrics
   ┌──────────┐   Utterance  ┌──┴───────┐   ┌───┴────────┐
   │Ingestion │─────────────▶│ Pipeline │──▶│ Reporting  │
   │ Azure /  │              └──┬────┬───┘   └────────────┘
   │simulated │     Detection   │    │  Guidance + Outcome
   └──────────┘   ┌─────────────▼┐  ┌▼──────────────┐  ┌──────────────┐
                  │  Classifier  │  │  RAG policy    │  │  Evidence    │
                  │ (DistilBERT, │  │  retriever     │  │  store       │
                  │  4 drivers)  │  │ (FAISS+MiniLM) │  │ append-only  │
                  └──────────────┘  └────────────────┘  └──────────────┘
```

Detection → guidance → handler action are all written to the immutable evidence
log, which feeds portfolio-level reporting.

## Project layout

```
src/vca/
  config.py        # YAML + env config
  schemas.py       # shared pydantic models (Utterance, Detection, Guidance, ...)
  ingestion/       # Azure Speech + simulated streaming sources
  classifier/      # fine-tuned multi-label driver classifier (train + infer)
  rag/             # policy indexing + adaptation retrieval
  guidance/        # builds the advisory prompt
  evidence/        # append-only hash-chained evidence store
  reporting/       # portfolio-level fair-treatment metrics
  pipeline.py      # wires the stages together
app/               # streamlit_app.py, api.py
data/              # synthetic transcripts, labeled data, firm policy
scripts/           # train_classifier, build_index, run_demo, generate_data
tests/
```

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 1. Generate synthetic data + mock policy (if not already present)
python scripts/generate_data.py

# 2. Fine-tune the driver classifier
python scripts/train_classifier.py

# 3. Build the policy RAG index
python scripts/build_index.py

# 4a. Run the end-to-end CLI demo over a sample conversation
python scripts/run_demo.py

# 4b. Or launch the web UI
streamlit run app/streamlit_app.py
```

Real Azure Speech is used when `AZURE_SPEECH_KEY` / `AZURE_SPEECH_REGION` are set
in `.env` (copy `.env.example`). Otherwise transcription falls back to the
simulated source. Set `VCA_FORCE_SIMULATED=1` to always use the fallback.

## Compliance & design notes

- **Advisory only.** The system never suspends collections, opens a case, or
  takes any customer-facing action. It surfaces guidance; a human decides.
- **Auditable.** The evidence log is append-only and hash-chained: each record
  embeds the previous record's hash, so tampering is detectable.
- **Evidenced fair treatment.** Portfolio reporting turns a judgement-dependent
  process into a demonstrable one.

See [docs/](docs/) for the full design and compliance write-up.
