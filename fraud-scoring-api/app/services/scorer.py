"""
Scorer service — loads pkl artifacts at startup, assembles features, runs inference.
Single instance shared across all FastAPI workers via module-level singleton.
"""
import time
t0 = time.perf_counter()
import os
import random
import pickle
import json
import logging
import numpy as np
from pathlib import Path

from app.services.divergence import divergence_log

logger = logging.getLogger("scorer")

# ── Artifact paths — relative to this file ────────────────────────────────────
_MODEL_DIR = Path(__file__).parent.parent / "models"
_CHAMPION_MODEL_PATH = _MODEL_DIR / "champion_model.pkl"
_CHALLENGER_MODEL_PATH = _MODEL_DIR / "challenger_model.pkl"
_SCALER_PATH = _MODEL_DIR / "scaler.pkl"
_META_PATH = _MODEL_DIR / "meta.json"

# ── Feature order must match train.py exactly ─────────────────────────────────
FEATURES = [
    "amount", "hour", "day_of_week",
    "cat_electronics", "cat_online", "cat_atm", "cat_travel",
    "cat_grocery", "cat_fuel", "cat_restaurant", "cat_pharmacy"
]

CATEGORIES = ["electronics", "online", "atm", "travel",
               "grocery", "fuel", "restaurant", "pharmacy"]


class Scorer:
    """
    Loads champion (LightGBM) + challenger (XGBoost) models, shared scaler, and meta
    once at startup. Both models score every request; the routed model's probability
    drives the decision, and the champion/challenger delta is logged for comparison.
    """

    def __init__(self):
        # Load artifacts — fail fast if missing (forces train.py to be run first)
        with open(_CHAMPION_MODEL_PATH, "rb") as f:
            self.champion_model = pickle.load(f)
        with open(_CHALLENGER_MODEL_PATH, "rb") as f:
            self.challenger_model = pickle.load(f)
        with open(_SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)
        with open(_META_PATH) as f:
            meta = json.load(f)

        champion_meta = meta["champion"]
        challenger_meta = meta["challenger"]

        # `threshold` stays the champion's — this is what's used for the
        # health endpoint and as the default decision boundary.
        self.threshold: float = champion_meta["threshold"]
        self.champion_threshold: float = champion_meta["threshold"]
        self.challenger_threshold: float = challenger_meta["threshold"]
        self.feature_importances: dict = champion_meta.get("feature_importances", {})
        self.roc_auc: float = champion_meta.get("roc_auc", 0.0)

        # Fraction of traffic routed to the challenger's score for the live decision.
        self.challenger_ratio: float = float(
            os.getenv("CHALLENGER_RATIO", meta.get("challenger_ratio_default", 0.1))
        )

        logger.info(
            f"Scorer loaded — champion(lgbm) threshold={self.champion_threshold:.3f} "
            f"auc={self.roc_auc:.3f} | challenger(xgb) threshold={self.challenger_threshold:.3f} "
            f"auc={challenger_meta.get('roc_auc', 0):.3f} | challenger_ratio={self.challenger_ratio:.2f}"
        )

    def score(self, amount, hour, day_of_week, merchant_category, velocity):
        cat_vec = [int(merchant_category == c) for c in CATEGORIES]
        feature_vec = np.array([[amount, hour, day_of_week] + cat_vec], dtype=float)
        feature_vec_scaled = self.scaler.transform(feature_vec)

        t0 = time.perf_counter()   # ← start timer HERE, after scaling
        champion_prob = float(self.champion_model.predict_proba(feature_vec_scaled)[0][1])
        challenger_prob = float(self.challenger_model.predict_proba(feature_vec_scaled)[0][1])
        latency = round((time.perf_counter() - t0) * 1000, 2)

        if random.random() < self.challenger_ratio:
            prob, threshold, model_used = challenger_prob, self.challenger_threshold, "challenger_xgb"
        else:
            prob, threshold, model_used = champion_prob, self.champion_threshold, "champion_lgbm"

        decision = "BLOCK" if prob >= threshold else "ALLOW"
        signals  = self._build_signals(amount, velocity, prob)

        divergence_log.record(champion_prob, challenger_prob, model_used)

        return {
        "fraud_probability": round(prob, 4),
        "champion_probability": round(champion_prob, 4),
        "challenger_probability": round(challenger_prob, 4),
        "model_used": model_used,
        "decision": decision,
        "threshold_used": round(threshold, 3),
        "top_signals": signals,
        "latency_ms": latency,
        }

    def _build_signals(self, amount: float, velocity: dict, prob: float) -> list[str]:
        """
        Human-readable signal strings ranked by severity.
        Mirrors the top-3 format shown in the README sample response.
        """
        signals = []

        zscore = velocity.get("amount_zscore", 0)
        # Clamp zscore display to ±10 to avoid overflow display artifact
        zscore_display = max(-10.0, min(10.0, zscore))
        if abs(zscore) >= 2:
            signals.append(
                f"amount_zscore={zscore_display:.2f} (≥2σ deviation from card history)"
            )

        txn_1hr = velocity.get("txn_count_1hr", 0)
        if txn_1hr >= 5:
            signals.append(f"txn_count_1hr={txn_1hr} (high velocity burst)")

        merchants_1hr = velocity.get("unique_merchants_1hr", 0)
        if merchants_1hr >= 4:
            signals.append(f"unique_merchants_1hr={merchants_1hr} (card-testing pattern)")

        time_since = velocity.get("time_since_last_txn_sec")
        if time_since is not None and time_since < 30:
            signals.append(f"time_since_last_txn_sec={time_since:.1f} (rapid repeat transaction)")

        # Always return at most top 3; pad with prob signal if nothing triggered
        if not signals:
            signals.append(f"fraud_probability={prob:.4f} (model score above threshold)")

        return signals[:3]


# Module-level singleton — loaded once, reused across all requests
scorer = Scorer()
# Alias for backward compatibility with score.py import
fraud_scorer = scorer
