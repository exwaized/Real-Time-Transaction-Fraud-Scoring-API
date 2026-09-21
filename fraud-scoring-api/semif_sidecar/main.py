"""
SemIf sidecar — loads the SemIf model once at startup and serves it over HTTP.

Kept as its own service on purpose: SemIf (github.com/TheoLeeCJ/SemIf) needs a
CUDA GPU and a resident multi-GB transformer, which is a different deployment
shape than the CPU-only sklearn scorer the main fraud API runs. Point
fraud-scoring-api's JEV_ENDPOINT at this service's /decide route and set
JEV_API_KEY / SEMIF_API_KEY to the same value on both sides.

Run: uvicorn main:app --host 0.0.0.0 --port 8600
"""
import os
import logging
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from semif_phase1.core import load_causal_model
from semif_phase1 import direct

logger = logging.getLogger("semif_sidecar")

# Pinned to a specific commit / model revision on purpose — SemIf is a
# community project with no stability guarantees; re-validate calibration
# before bumping either of these.
MODEL_ID = os.getenv("SEMIF_MODEL", "Qwen/Qwen3.5-4B")
MODEL_REVISION = os.getenv("SEMIF_MODEL_REVISION", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
MAX_TOKENS = int(os.getenv("SEMIF_MAX_TOKENS", "4096"))
API_KEY = os.getenv("SEMIF_API_KEY")

ACTION_DESCRIPTIONS = {
    "ALLOW": "Let the transaction proceed with no further action.",
    "BLOCK": "Reject the transaction as fraudulent.",
    "REVIEW": "Hold the transaction for manual analyst review.",
}

app = FastAPI(title="SemIf sidecar")

_model = None
_tokenizer = None
_metadata = None


@app.on_event("startup")
def load_model():
    global _model, _tokenizer, _metadata
    logger.info(f"Loading {MODEL_ID}@{MODEL_REVISION} ...")
    _model, _tokenizer, _metadata = load_causal_model(MODEL_ID, MODEL_REVISION)
    logger.info("SemIf model loaded")


class DecideRequest(BaseModel):
    situation: dict
    allowed_actions: list[str]


def _check_auth(authorization: Optional[str]):
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="invalid or missing API key")


@app.post("/decide")
def decide(body: DecideRequest, authorization: Optional[str] = Header(default=None)):
    """
    Mirrors the payload adjudicator.py's _call_jev() already sends:
    {"situation": {...}, "allowed_actions": [...]}
    Returns {"decision": <one of allowed_actions>, "reasoning": <str>}.
    """
    _check_auth(authorization)

    if _model is None:
        raise HTTPException(status_code=503, detail="model still loading")

    row = {
        "id": "fraud-adjudication",
        "state": body.situation,
        "question": "What should happen to this transaction?",
        "options": [
            {"id": action, "description": ACTION_DESCRIPTIONS.get(action, action)}
            for action in body.allowed_actions
        ],
    }

    result = direct.score(_model, _tokenizer, row, _metadata, MAX_TOKENS)

    probs = dict(zip(result["option_ids"], result["probabilities"]))
    decision = max(probs, key=probs.get)
    reasoning = "SemIf probabilities: " + ", ".join(f"{k}={v:.3f}" for k, v in probs.items())

    return {"decision": decision, "reasoning": reasoning}


@app.get("/health")
def health():
    return {"status": "ok" if _model is not None else "loading", "model": MODEL_ID}
