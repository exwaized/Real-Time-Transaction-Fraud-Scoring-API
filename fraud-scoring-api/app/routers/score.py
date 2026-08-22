import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from app.models.schemas import TransactionRequest, FraudScoreResponse, HealthResponse
from app.services.velocity import velocity_engine
from app.services.scorer import fraud_scorer
from app.services.drift import drift_monitor
from app.services.auth import verify_api_key
from app.services.rate_limit import rate_limiter
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
        ts = datetime.fromisoformat(txn.timestamp) if txn.timestamp else datetime.now()
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

        logger.info(
            f"SCORED card={txn.card_id} amount={txn.amount} "
            f"prob={result['fraud_probability']} decision={result['decision']} "
            f"latency={result['latency_ms']}ms"
        )

        return FraudScoreResponse(
            card_id=txn.card_id,
            **result,
            velocity=velocity,
            behavioral_profile=profile,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Scoring error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse, dependencies=[Depends(protected)])
def health():
    return HealthResponse(
        status="ok",
        model_threshold=fraud_scorer.threshold,
        drift_status=drift_monitor.get_status(),
        uptime_sec=round(time.time() - START_TIME, 1),
    )


@router.get("/model/info", dependencies=[Depends(protected)])
def model_info():
    import json
    from pathlib import Path
    meta_path = Path(__file__).parent.parent / "models" / "meta.json"
    with open(meta_path) as f:
        return json.load(f)
