"""GuardianCX FastAPI backend.

Exposes the agent pipeline as a service so the detection/guidance/evidence
capabilities can be consumed by other channels (IVR, chat backends, QA tooling)
in addition to the Streamlit console.

Run:  uvicorn api:app --reload   (from the guardiancx/ directory)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from guardiancx.agents.graph import process_turn  # noqa: E402
from guardiancx.database.db import init_engine  # noqa: E402
from guardiancx.database.repository import (  # noqa: E402
    list_evidence,
    list_pending,
    update_approval,
    verify_chain,
)
from guardiancx.rag.vector_store import ensure_ingested, get_vector_store  # noqa: E402
from guardiancx.utils.types import Driver  # noqa: E402

init_engine()
ensure_ingested()
app = FastAPI(title="GuardianCX", version="0.1.0")


class TurnRequest(BaseModel):
    conversation_id: str = "API"
    customer_id: str = ""
    turn_index: int = 0
    speaker: str = "customer"
    text: str


class ApprovalRequest(BaseModel):
    record_id: str
    status: str  # approved | rejected
    decided_by: str
    note: str = ""


@app.post("/assess")
def assess(req: TurnRequest):
    """Run one turn through the multi-agent pipeline; returns the decision."""
    state = process_turn(req.conversation_id, req.customer_id, req.turn_index,
                         req.speaker, req.text)
    d = state["decision"]
    return {
        "masked_text": state.get("masked_text", ""),
        "drivers": [x.value for x in d.assessment.triggered],
        "detection_source": d.assessment.source,
        "risk_level": d.risk_level.value,
        "approval_status": d.approval_status.value,
        "recommendation": d.recommendation.model_dump() if d.recommendation else None,
        "guardrails": [r.model_dump() for r in d.guardrails.results],
        "record_id": state.get("record_id", ""),
        "trace": state.get("trace", []),
    }


@app.post("/policy/search")
def policy_search(q: str, driver: str | None = None, top_k: int = 3):
    d = Driver(driver) if driver else None
    return [c.model_dump() for c in get_vector_store().query(q, driver=d, top_k=top_k)]


@app.get("/pending")
def pending():
    return list_pending()


@app.post("/approve")
def approve(req: ApprovalRequest):
    ok = update_approval(req.record_id, req.status, req.decided_by, req.note)
    return {"updated": ok}


@app.get("/evidence")
def evidence(limit: int = 100):
    return list_evidence(limit)


@app.get("/audit/verify")
def audit_verify():
    ok, bad = verify_chain()
    return {"chain_valid": ok, "first_bad_record": bad}
