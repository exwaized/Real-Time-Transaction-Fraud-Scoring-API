"""
Adjudicator sidecar — loads a Laya decision checkpoint once at startup and
serves it over the /decide contract app/services/adjudicator.py already speaks.

Backend: Laya (github.com/NandhaKishorM/laya, Apache-2.0), a small (~421M
param) non-autoregressive encoder that answers a fixed set of typed
questions (choice/score/noul) in one forward pass and returns a calibrated
probability per option. Runs CPU-only in well under a second per call, which
is what let this move off the GPU-gated SemIf/Qwen3.5-4B setup this service
used to run — see docker-compose.yml history for that version.

Laya ships its own HTTP server (`laya.serve`, `pip install laya[serve]`) but
that speaks TypeSafe Jev's `/v1/systemone` wire protocol ({state, questions}
in, {model, answers, usage, routing} out) — a different shape than
adjudicator.py's `_call_jev()` ({situation, allowed_actions} in,
{decision, reasoning} out). Rather than chain two HTTP hops (this service ->
laya-serve -> Laya), this loads Laya's `Router` directly in-process and
translates the request/response shape itself; adjudicator.py needed no
changes at all.

Run: uvicorn main:app --host 0.0.0.0 --port 8600
"""
import os
import logging
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

import laya

logger = logging.getLogger("semif_sidecar")

# The three checkpoints Laya ships (see laya.router.DEFAULT_MODELS) are
# "english", "multilingual" and "typed-decisions" — the last is the one
# trained specifically for fixed-option decisions like ours, so it's the
# default here rather than "english" (Laya's own router default).
LAYA_CHECKPOINT = os.getenv("LAYA_CHECKPOINT", "typed-decisions")
LAYA_DEVICE = os.getenv("LAYA_DEVICE") or None
API_KEY = os.getenv("SEMIF_API_KEY")

# Caps torch's intra-op thread count for CPU inference. Left unset by default
# (torch's own default); set to <= the container's physical core count if
# oversubscribing logical/hyperthread count causes a latency regression under
# concurrent requests (see laya.serve's docstring, which flags the same knob).
LAYA_THREADS = os.getenv("LAYA_THREADS")

ACTION_DESCRIPTIONS = {
    "ALLOW": "Let the transaction proceed with no further action.",
    "BLOCK": "Reject the transaction as fraudulent.",
    "REVIEW": "Hold the transaction for manual analyst review.",
}

app = FastAPI(title="Adjudicator sidecar (Laya)")

_router: Optional["laya.Router"] = None


@app.on_event("startup")
def load_model():
    global _router
    if LAYA_THREADS:
        try:
            import torch
            torch.set_num_threads(int(LAYA_THREADS))
        except (ValueError, ImportError):
            logger.warning(f"Ignoring invalid LAYA_THREADS={LAYA_THREADS!r}")

    logger.info(f"Loading Laya checkpoint '{LAYA_CHECKPOINT}' ...")
    _router = laya.Router(device=LAYA_DEVICE, preload=False)
    _router.load(LAYA_CHECKPOINT)
    logger.info("Laya checkpoint loaded")


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

    if _router is None:
        raise HTTPException(status_code=503, detail="model still loading")

    questions = {
        "decision": {
            "type": "choice",
            "instructions": "Given the transaction situation, what should happen to this transaction?",
            "criteria": {
                action: ACTION_DESCRIPTIONS.get(action, action)
                for action in body.allowed_actions
            },
        }
    }

    result = _router.predict(body.situation, questions, model=LAYA_CHECKPOINT)
    answer = result["answers"]["decision"]
    decision = answer["choice"]
    reasoning = "Laya probabilities: " + ", ".join(
        f"{k}={v:.3f}" for k, v in answer["probabilities"].items()
    )

    return {"decision": decision, "reasoning": reasoning}


@app.get("/health")
def health():
    loaded = _router is not None and LAYA_CHECKPOINT in _router.loaded
    return {"status": "ok" if loaded else "loading", "model": LAYA_CHECKPOINT}
