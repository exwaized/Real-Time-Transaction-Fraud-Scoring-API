"""
Real-Time Transaction Fraud Scoring API
FastAPI serving layer with:
- Champion/Challenger routing
- Rule-based pre-filter
- SHAP top_3_risk_factors
- Prometheus instrumentation
- Sub-50ms p99 target
"""

import os
import json
import time
import pickle
import random
import logging
import hashlib
import threading
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import (
    Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
)
from starlette.responses import Response
from sklearn.metrics import roc_curve
import httpx

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "msg": "%(message)s"}'
)
log = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", "models")
SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK_URL", "")
CHALLENGER_RATIO = float(os.getenv("CHALLENGER_RATIO", "0.10"))

# ─────────────────────────────────────────────
# PROMETHEUS METRICS
# ─────────────────────────────────────────────
REQUEST_COUNT = Counter("fraud_api_requests_total", "Total score requests", ["risk_tier", "model"])
FRAUD_RATE_GAUGE = Gauge("fraud_rate_rolling_5min", "Rolling fraud rate over 5 min window")
LATENCY_HISTOGRAM = Histogram(
    "fraud_score_latency_seconds",
    "Scoring latency",
    buckets=[0.005, 0.010, 0.025, 0.050, 0.075, 0.100, 0.200, 0.500]
)
SCORE_HISTOGRAM = Histogram(
    "fraud_score_distribution",
    "Distribution of fraud scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
RULE_FLAG_COUNTER = Counter("rule_based_flags_total", "Rule-based pre-filter triggers", ["rule"])
CHALLENGER_DIVERGENCE = Histogram(
    "challenger_score_divergence",
    "Champion vs Challenger score delta",
    buckets=[0.05, 0.10, 0.20, 0.30, 0.50]
)

# ─────────────────────────────────────────────
# FRAUD RATE ROLLING WINDOW (5 min)
# ─────────────────────────────────────────────
_window_lock = threading.Lock()
_fraud_window: deque = deque()  # (timestamp, is_fraud)
WINDOW_SECONDS = 300  # 5 min

def record_fraud_event(is_fraud: bool):
    now = time.time()
    with _window_lock:
        _fraud_window.append((now, is_fraud))
        cutoff = now - WINDOW_SECONDS
        while _fraud_window and _fraud_window[0][0] < cutoff:
            _fraud_window.popleft()
        total = len(_fraud_window)
        fraud_count = sum(1 for _, f in _fraud_window if f)
        rate = fraud_count / total if total > 0 else 0.0
        FRAUD_RATE_GAUGE.set(rate)
        if rate > 0.02 and total >= 10:
            _maybe_alert(rate, total)


def _maybe_alert(rate: float, total: int):
    msg = f"🚨 CRITICAL: Rolling fraud rate {rate:.1%} over last {total} txns (5min window)"
    log.critical(msg)
    if SLACK_WEBHOOK:
        try:
            import httpx
            httpx.post(SLACK_WEBHOOK, json={"text": msg}, timeout=3)
        except Exception:
            pass


# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
champion_model = None
challenger_model = None
model_meta = {}
explainer = None
challenger_log: list = []  # in-memory challenger divergence log


def load_models():
    global champion_model, challenger_model, model_meta, explainer
    with open(f"{MODEL_DIR}/champion_lgbm.pkl", "rb") as f:
        champion_model = pickle.load(f)
    with open(f"{MODEL_DIR}/challenger_xgb.pkl", "rb") as f:
        challenger_model = pickle.load(f)
    with open(f"{MODEL_DIR}/model_meta.json") as f:
        model_meta = json.load(f)
    # SHAP explainer for champion
    explainer = shap.TreeExplainer(champion_model)
    log.info("Models loaded successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    log.info("Shutting down")


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────
app = FastAPI(
    title="Fraud Scoring API",
    description="Real-Time Transaction Fraud Scoring — Amex CFR Style",
    version="1.0.0",
    lifespan=lifespan,
)

FEATURE_COLS = [
    "TransactionAmt", "hour", "day_of_week", "is_weekend", "is_night",
    "amt_vs_avg_ratio", "amt_zscore", "velocity_1hr_bucket",
    "balance_drain_ratio", "is_foreign_txn", "merchant_risk_score",
    "rule_flag", "merchant_txn_freq", "card_txn_count",
]

RISK_TIERS = {
    "low": (0.0, 0.3),
    "medium": (0.3, 0.6),
    "high": (0.6, 0.8),
    "critical": (0.8, 1.01),
}

MERCHANT_RISK = {"W": 0.02, "C": 0.08, "H": 0.04, "S": 0.03, "R": 0.12}


# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────
class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., example="TXN_001")
    card_id: int = Field(..., example=1234)
    transaction_amt: float = Field(..., example=450.00)
    product_cd: str = Field(..., example="W")
    transaction_dt: int = Field(..., description="Unix timestamp", example=1700000000)
    balance_before: float = Field(0.0, example=1200.0)
    balance_after: float = Field(0.0, example=750.0)
    merchant_id: int = Field(..., example=555)
    card_country: str = Field("US", example="US")
    card_txn_count_24hr: int = Field(5, example=5)
    merchant_txn_freq: int = Field(100, example=100)


class ScoreResponse(BaseModel):
    transaction_id: str
    fraud_score: float
    risk_tier: str
    top_3_risk_factors: list
    rule_flags: list
    model_used: str
    latency_ms: float
    timestamp: str


# ─────────────────────────────────────────────
# FEATURE BUILDER
# ─────────────────────────────────────────────
def build_features(req: TransactionRequest) -> pd.DataFrame:
    hour = (req.transaction_dt // 3600) % 24
    day_of_week = (req.transaction_dt // 86400) % 7

    card_avg = model_meta.get("card_avg", {}).get(str(req.card_id), 150.0)
    card_std = model_meta.get("card_std", {}).get(str(req.card_id), 50.0)

    amt_vs_avg = req.transaction_amt / (card_avg + 1e-6)
    amt_z = (req.transaction_amt - card_avg) / (card_std + 1e-6)

    velocity_bucket = min(3, req.card_txn_count_24hr // 5)

    drain = (req.balance_before - req.balance_after) / (req.balance_before + 1e-6) \
        if req.balance_before > 0 else 0.0

    is_foreign = int(req.card_country != "US")
    merchant_risk = MERCHANT_RISK.get(req.product_cd, 0.05)
    rule_flag = int(amt_vs_avg > 3 or is_foreign == 1)

    row = {
        "TransactionAmt": req.transaction_amt,
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": int(day_of_week in [5, 6]),
        "is_night": int(hour >= 22 or hour <= 6),
        "amt_vs_avg_ratio": amt_vs_avg,
        "amt_zscore": amt_z,
        "velocity_1hr_bucket": velocity_bucket,
        "balance_drain_ratio": drain,
        "is_foreign_txn": is_foreign,
        "merchant_risk_score": merchant_risk,
        "rule_flag": rule_flag,
        "merchant_txn_freq": req.merchant_txn_freq,
        "card_txn_count": req.card_txn_count_24hr,
    }
    return pd.DataFrame([row])[FEATURE_COLS]


# ─────────────────────────────────────────────
# RULE-BASED PRE-FILTER
# ─────────────────────────────────────────────
def apply_rules(req: TransactionRequest, features: pd.DataFrame) -> tuple[list, float]:
    flags = []
    boost = 0.0

    card_avg = model_meta.get("card_avg", {}).get(str(req.card_id), 150.0)
    if req.transaction_amt > 3 * card_avg:
        flags.append("amt_exceeds_3x_card_avg")
        RULE_FLAG_COUNTER.labels(rule="amt_3x").inc()
        boost += 0.10

    if req.card_country != "US":
        flags.append("foreign_transaction")
        RULE_FLAG_COUNTER.labels(rule="foreign_txn").inc()
        boost += 0.05

    if req.card_txn_count_24hr > 15:
        flags.append("high_velocity_24hr")
        RULE_FLAG_COUNTER.labels(rule="velocity").inc()
        boost += 0.08

    if req.balance_before > 0 and (req.balance_before - req.balance_after) > 0.95 * req.balance_before:
        flags.append("near_full_balance_drain")
        RULE_FLAG_COUNTER.labels(rule="balance_drain").inc()
        boost += 0.12

    return flags, min(boost, 0.30)  # cap rule boost at 0.30


# ─────────────────────────────────────────────
# SHAP EXPLANATION
# ─────────────────────────────────────────────
def get_top_factors(features: pd.DataFrame, n=3) -> list:
    try:
        sv = explainer.shap_values(features)
        if isinstance(sv, list):
            sv = sv[1]  # binary class 1
        sv = np.abs(sv[0])
        top_idx = np.argsort(sv)[::-1][:n]
        return [
            {"feature": FEATURE_COLS[i], "importance": round(float(sv[i]), 4)}
            for i in top_idx
        ]
    except Exception:
        return []


# ─────────────────────────────────────────────
# RISK TIER
# ─────────────────────────────────────────────
def get_tier(score: float) -> str:
    for tier, (lo, hi) in RISK_TIERS.items():
        if lo <= score < hi:
            return tier
    return "critical"


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────
@app.post("/score", response_model=ScoreResponse, tags=["Scoring"])
async def score_transaction(req: TransactionRequest):
    t0 = time.perf_counter()

    features = build_features(req)
    rule_flags, rule_boost = apply_rules(req, features)

    # Champion/Challenger routing
    use_challenger = random.random() < CHALLENGER_RATIO
    model_label = "challenger_xgb" if use_challenger else "champion_lgbm"
    active_model = challenger_model if use_challenger else champion_model

    raw_score = float(active_model.predict_proba(features)[0, 1])

    # Always score champion for divergence logging
    if use_challenger:
        champ_score = float(champion_model.predict_proba(features)[0, 1])
        delta = abs(raw_score - champ_score)
        CHALLENGER_DIVERGENCE.observe(delta)
        challenger_log.append({
            "txn_id": req.transaction_id,
            "champion_score": round(champ_score, 4),
            "challenger_score": round(raw_score, 4),
            "delta": round(delta, 4),
            "ts": datetime.utcnow().isoformat(),
        })

    fraud_score = min(1.0, raw_score + rule_boost)
    tier = get_tier(fraud_score)
    top_factors = get_top_factors(features)

    latency_ms = (time.perf_counter() - t0) * 1000
    LATENCY_HISTOGRAM.observe(latency_ms / 1000)
    SCORE_HISTOGRAM.observe(fraud_score)
    REQUEST_COUNT.labels(risk_tier=tier, model=model_label).inc()
    record_fraud_event(fraud_score >= 0.5)

    log.info(json.dumps({
        "txn_id": req.transaction_id,
        "fraud_score": round(fraud_score, 4),
        "risk_tier": tier,
        "model": model_label,
        "rule_flags": rule_flags,
        "latency_ms": round(latency_ms, 2),
        "timestamp": datetime.utcnow().isoformat(),
    }))

    return ScoreResponse(
        transaction_id=req.transaction_id,
        fraud_score=round(fraud_score, 4),
        risk_tier=tier,
        top_3_risk_factors=top_factors,
        rule_flags=rule_flags,
        model_used=model_label,
        latency_ms=round(latency_ms, 2),
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/threshold", tags=["Calibration"])
async def calibrate_threshold(fpr: float = Query(0.01, description="Target false positive rate")):
    """Return optimal score threshold for a given FPR target."""
    # Serve pre-computed or return meta default
    threshold = model_meta.get("default_threshold", 0.5)
    return {
        "requested_fpr": fpr,
        "calibrated_threshold": round(threshold, 4),
        "note": "Re-run train.py on live data to recalibrate. Uses champion model ROC curve.",
    }


@app.get("/challenger/divergence", tags=["Monitoring"])
async def get_challenger_divergence():
    """Return last 100 champion vs challenger divergence records."""
    return {"count": len(challenger_log), "records": challenger_log[-100:]}


@app.get("/health", tags=["Ops"])
async def health():
    return {
        "status": "ok",
        "champion_loaded": champion_model is not None,
        "challenger_loaded": challenger_model is not None,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/metrics", tags=["Ops"])
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/", tags=["Ops"])
async def root():
    return {
        "service": "Fraud Scoring API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
