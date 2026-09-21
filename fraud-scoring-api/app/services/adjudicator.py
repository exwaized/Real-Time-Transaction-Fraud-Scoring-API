"""
Adjudicator — confidence-gated review step, inserted between the scorer's raw
probability and the final ALLOW/BLOCK decision.

Why this exists: fraud_scorer.score() applies a flat threshold (scorer.py) —
every transaction gets a binary call regardless of how close its probability
sits to that threshold. Most transactions score far from the threshold and
are correctly confident either way. Only the narrow band around the threshold
is genuinely ambiguous, and that's the only slice worth paying for a second,
more expensive opinion on.

This module owns exactly that gate:
  - needs_review()  decides whether a transaction falls in the ambiguous band
  - review()        is the (pluggable) second-opinion call for that band only

Wiring a real adjudicator (Jev, an LLM, whatever) means filling in
_call_jev() below with its actual endpoint/payload/auth once that's known —
the gate and fail-open behavior around it don't need to change.
"""
import os
import logging
import httpx

logger = logging.getLogger("adjudicator")

# How close to the model's threshold counts as "ambiguous enough to review".
# 0.05 means a threshold of 0.80 reviews probabilities in [0.75, 0.85].
REVIEW_MARGIN = float(os.getenv("REVIEW_MARGIN", "0.05"))

JEV_ENDPOINT = os.getenv("JEV_ENDPOINT")
JEV_API_KEY = os.getenv("JEV_API_KEY")


class Adjudicator:
    """
    Confidence-gated second opinion for borderline fraud scores.
    Disabled (fail-open to the raw threshold decision) unless JEV_ENDPOINT /
    JEV_API_KEY are configured, so the pipeline behaves exactly as before
    until this is actually wired up.
    """

    def __init__(self, margin: float = REVIEW_MARGIN):
        self.margin = margin
        self.enabled = bool(JEV_ENDPOINT and JEV_API_KEY)
        if not self.enabled:
            logger.info(
                "Adjudicator disabled — set JEV_ENDPOINT and JEV_API_KEY to enable "
                "review of borderline scores (margin=%.3f)", self.margin
            )

    def needs_review(self, prob: float, threshold: float) -> bool:
        return abs(prob - threshold) < self.margin

    def review(self, amount: float, velocity: dict, profile: dict,
               signals: list[str], prob: float, threshold: float) -> dict:
        """
        Second opinion for a borderline transaction.
        Returns {"decision": "ALLOW"|"BLOCK"|"REVIEW", "reasoning": str, "reviewed": bool}.
        Never raises — any failure falls back to the original threshold decision
        so an outage here can't take the scoring endpoint down with it.
        """
        fallback_decision = "BLOCK" if prob >= threshold else "ALLOW"

        if not self.enabled:
            return {"decision": fallback_decision, "reasoning": None, "reviewed": False}

        try:
            decision, reasoning = self._call_jev(
                amount=amount, velocity=velocity, profile=profile,
                signals=signals, prob=prob, threshold=threshold,
            )
            return {"decision": decision, "reasoning": reasoning, "reviewed": True}
        except Exception as e:
            logger.warning(f"Adjudicator call failed, falling back to threshold decision: {e}")
            return {"decision": fallback_decision, "reasoning": None, "reviewed": False}

    def _call_jev(self, amount, velocity, profile, signals, prob, threshold) -> tuple[str, str]:
        """
        Actual network call to the adjudicator.

        TODO: this payload/response shape is a placeholder — Jev's real
        request/response schema isn't wired in yet. Replace once you have
        their API docs; everything above this method (the gate, the fail-open
        fallback, the response shape returned to score.py) stays the same.
        """
        payload = {
            "situation": {
                "amount": amount,
                "velocity": velocity,
                "behavioral_profile": profile,
                "model_signals": signals,
                "model_probability": prob,
                "model_threshold": threshold,
            },
            "allowed_actions": ["ALLOW", "BLOCK", "REVIEW"],
        }
        resp = httpx.post(
            JEV_ENDPOINT,
            json=payload,
            headers={"Authorization": f"Bearer {JEV_API_KEY}"},
            timeout=2.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["decision"], data.get("reasoning")


# Singleton — mirrors scorer.py / drift.py
adjudicator = Adjudicator()
