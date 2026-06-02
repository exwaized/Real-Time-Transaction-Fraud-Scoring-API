"""
Score router — all three endpoints live here.
Wires together VelocityEngine, Scorer, and DriftMonitor into the request lifecycle.
"""

import time
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException

# Import singletons — all three are module-level instances, loaded once at startup
from app.models.schemas import TransactionRequest, ScoreResponse, VelocityFeatures, BehavioralProfile
from app.services.scorer import scorer
from velocity import velocity_engine
from drift import drift_monitor

logger = logging.getLogger("score_router")
router = APIRouter()


@router.post("/score", response_model=ScoreResponse)
def score_transaction(txn: TransactionRequest):
    """
    Main scoring endpoint.
    1. Record transaction into VelocityEngine (updates rolling window + profile)
    2. Compute velocity features for this card
    3. Run Scorer to get fraud probability + decision
    4. Feed score + features into DriftMonitor for background monitoring
    5. Return full response
    """
    t0 = time.perf_counter()

    # Step 1 — record into rolling window BEFORE computing features
    # so the current txn is included in velocity counts
    velocity_engine.record(
        card_id=txn.card_id,
        amount=txn.amount,
        merchant_id=txn.merchant_id,
        category=txn.merchant_category,
    )

    # Step 2 — pull velocity features (RLock-protected, O(n) over small window)
    velocity_feats = velocity_engine.get_velocity_features(txn.card_id, txn.amount)
    profile = velocity_engine.get_profile(txn.card_id)

    # Step 3 — run model inference
    hour = txn.timestamp.hour
    day_of_week = txn.timestamp.weekday()
    result = scorer.score(
        amount=txn.amount,
        hour=hour,
        day_of_week=day_of_week,
        merchant_category=txn.merchant_category,
        velocity=velocity_feats,
    )

    # Step 4 — feed into drift monitor (non-blocking, daemon thread handles it)
    drift_monitor.record(score=result["fraud_probability"], features=velocity_feats)

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        f"card={txn.card_id} prob={result['fraud_probability']:.4f} "
        f"decision={result['decision']} latency={latency_ms:.2f}ms"
    )

    return ScoreResponse(
        card_id=txn.card_id,
        fraud_probability=result["fraud_probability"],
        decision=result["decision"],
        threshold_used=result["threshold_used"],
        top_signals=result["top_signals"],
        velocity=VelocityFeatures(**velocity_feats),
        behavioral_profile=BehavioralProfile(
            avg_amount=profile.get("avg_amount", txn.amount),
            top_hour=profile.get("top_hour"),
            top_category=profile.get("top_category"),
            total_txns_seen=profile.get("total_txns_seen", 1),
        ),
        latency_ms=round(latency_ms, 2),
    )


@router.get("/health")
def health():
    """
    Returns model load status + current drift monitor state.
    Use this to confirm the service is live and the daemon thread is running.
    """
    drift_status = drift_monitor.get_status()
    return {
        "status": "ok",
        "model_loaded": True,
        "threshold": scorer.threshold,
        "drift_monitor": drift_status,
    }


@router.get("/model/info")
def model_info():
    """
    Returns training metadata — threshold, ROC-AUC, feature importances.
    Useful for audit trails and verifying which model version is serving.
    """
    return {
        "threshold": scorer.threshold,
        "roc_auc": scorer.roc_auc,
        "features": scorer.feature_importances,
    }
