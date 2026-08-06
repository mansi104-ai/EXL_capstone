"""Minimal FastAPI service exposing the VCA pipeline.

Endpoints:
  POST /assess    -> classify one utterance and return advisory guidance (no
                     evidence is written; this is a stateless assessment).
  POST /record    -> classify, produce guidance, and append to the evidence log
                     with the supplied handler outcome.
  GET  /report     -> portfolio-level report + chain verification.

Run:  uvicorn app.api:app --reload
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vca.config import load_config  # noqa: E402
from vca.pipeline import VCAPipeline  # noqa: E402
from vca.reporting.metrics import build_report  # noqa: E402
from vca.schemas import HandlerAction, Outcome  # noqa: E402

cfg = load_config()
pipeline = VCAPipeline.from_config(cfg)
app = FastAPI(title="Vulnerable Customer Care Agent", version="0.1.0")


class AssessRequest(BaseModel):
    conversation_id: str = "CONV"
    turn_index: int = 0
    text: str


class RecordRequest(AssessRequest):
    handler_action: HandlerAction = HandlerAction.NO_RESPONSE
    note: str | None = None


@app.post("/assess")
def assess(req: AssessRequest):
    detection = pipeline.classifier.detect(req.conversation_id, req.turn_index, req.text)
    guidance = pipeline.advisor.advise(detection)
    return {
        "detection": detection.model_dump(mode="json"),
        "guidance": guidance.model_dump(mode="json") if guidance else None,
        "advisory_only": True,
    }


@app.post("/record")
def record(req: RecordRequest):
    detection = pipeline.classifier.detect(req.conversation_id, req.turn_index, req.text)
    guidance = pipeline.advisor.advise(detection)
    if guidance is None:
        return {"detection": detection.model_dump(mode="json"), "recorded": False}
    outcome = Outcome(action=req.handler_action, note=req.note)
    rec = pipeline.evidence.append(detection, guidance, outcome)
    return {"recorded": True, "record_id": rec.record_id, "record_hash": rec.record_hash}


@app.get("/report")
def report():
    return build_report(pipeline.evidence).as_dict()
