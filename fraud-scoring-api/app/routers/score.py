import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from app.models.schemas import TransactionRequest, FraudScoreResponse, HealthResponse
from app.services.velocity import velocity_engine
from app.services.scorer import fraud_scorer
from app.services.drift import drift_monitor
from app.services.auth import verify_api_key
from app.services.rate_limit import rate_limiter
from app.services.adjudicator import adjudicator
from app.services.divergence import divergence_log
from app.services.metrics import (
    REQUESTS, ERRORS, SCORE_DISTRIBUTION, CHALLENGER_DIVERGENCE, SIGNALS, ADJUDICATOR_FALLBACKS, SCORE_LATENCY, render as render_metrics,
)
from fastapi import Response
import time

router = APIRouter()
logger = logging.getLogger("router")
START_TIME = time.time()

# Reusable dependency: auth + rate limit together
def protected(api_key: str = Depends(verify_api_key)):
    rate_limiter.check(api_key)
    return api_key

@router.post("/score", response_model=FraudScoreResponse, dependencies=[Depends(protected)])
def score_transaction(txn: TransactionRequest):
    """
    Score a transaction for fraud.
    Requires X-API-Key header. Rate limited to 60 req/min per key.
    """
    try:
        _t0 = time.perf_counter()
        try:
            ts = datetime.fromisoformat(txn.timestamp) if txn.timestamp else datetime.now()
        except ValueError:
            raise HTTPException(status_code=422, detail="timestamp must be ISO 8601")
        hour        = ts.hour
        day_of_week = ts.weekday()

        # 1. Velocity features
        velocity = velocity_engine.get_velocity_features(txn.card_id, txn.amount)

        # 2. Score
        result = fraud_scorer.score(
            amount=txn.amount,
            hour=hour,
            day_of_week=day_of_week,
            merchant_category=txn.merchant_category,
            velocity=velocity,
        )

        # 3. Record AFTER scoring (don't inflate velocity on current txn)
        velocity_engine.record(txn.card_id, txn.amount, txn.merchant_id, txn.merchant_category)

        # 4. Feed drift monitor
        drift_monitor.record(result["fraud_probability"], velocity)

        # 5. Behavioral profile
        profile = velocity_engine.get_profile(txn.card_id)

        # 6. Borderline scores get a second opinion; everything else is untouched
        reviewed = False
        review_reasoning = None
        if adjudicator.needs_review(result["fraud_probability"], result["threshold_used"]):
            outcome = adjudicator.review(
                amount=txn.amount,
                velocity=velocity,
                profile=profile,
                signals=result["top_signals"],
                prob=result["fraud_probability"],
                threshold=result["threshold_used"],
            )
            result["decision"] = outcome["decision"]
            reviewed = outcome["reviewed"]
            review_reasoning = outcome["reasoning"]
            if not reviewed:
                ADJUDICATOR_FALLBACKS.inc()

        REQUESTS.labels(result["decision"], result["model_used"]).inc()
        SCORE_DISTRIBUTION.observe(result["fraud_probability"])
        CHALLENGER_DIVERGENCE.observe(
            abs(result["champion_probability"] - result["challenger_probability"]))
        for sig in result["top_signals"]:
            SIGNALS.labels(sig.split("=")[0]).inc()
        SCORE_LATENCY.observe(time.perf_counter() - _t0)

        logger.info(
            f"SCORED card={txn.card_id} amount={txn.amount} "
            f"prob={result['fraud_probability']} decision={result['decision']} "
            f"reviewed={reviewed} latency={result['latency_ms']}ms"
        )

        return FraudScoreResponse(
            card_id=txn.card_id,
            **result,
            velocity=velocity,
            behavioral_profile=profile,
            reviewed=reviewed,
            review_reasoning=review_reasoning,
        )

    except HTTPException:
        raise
    except Exception as e:
        ERRORS.inc()
        logger.error(f"Scoring error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal scoring error")


@router.get("/healthz")
def healthz():
    """Unauthenticated liveness probe for load balancers / docker healthcheck."""
    return {"status": "ok"}


@router.get("/metrics", include_in_schema=False)
def metrics():
    # Unauthenticated so Prometheus can scrape; expose only on the internal
    # network (reverse proxy should not route /metrics publicly).
    body, content_type = render_metrics()
    return Response(content=body, media_type=content_type)


@router.get("/health", response_model=HealthResponse, dependencies=[Depends(protected)])
def health():
    return HealthResponse(
        status="ok",
        model_threshold=fraud_scorer.threshold,
        drift_status=drift_monitor.get_status(),
        uptime_sec=round(time.time() - START_TIME, 1),
    )


@router.get("/challenger/divergence", dependencies=[Depends(protected)])
def challenger_divergence():
    """Last 100 champion vs challenger score delta records."""
    records = divergence_log.get_records()
    return {"records": records, "count": len(records)}


@router.get("/model/info", dependencies=[Depends(protected)])
def model_info():
    import json
    from pathlib import Path
    meta_path = Path(__file__).parent.parent / "models" / "meta.json"
    with open(meta_path) as f:
        return json.load(f)
