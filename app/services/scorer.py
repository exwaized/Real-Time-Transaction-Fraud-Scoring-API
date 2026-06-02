"""
Scorer service — loads pkl artifacts at startup, assembles features, runs inference.
Single instance shared across all FastAPI workers via module-level singleton.
"""

import pickle
import json
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger("scorer")

# ── Artifact paths — relative to this file ────────────────────────────────────
_MODEL_DIR = Path(__file__).parent.parent / "models"
_MODEL_PATH = _MODEL_DIR / "model.pkl"
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
    Loads model + scaler + meta once at startup.
    score() assembles the feature vector and returns probability + signals.
    """

    def __init__(self):
        # Load artifacts — fail fast if missing (forces train.py to be run first)
        with open(_MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        with open(_SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)
        with open(_META_PATH) as f:
            meta = json.load(f)

        self.threshold: float = meta["threshold"]
        self.feature_importances: dict = meta.get("feature_importances", {})
        self.roc_auc: float = meta.get("roc_auc", 0.0)
        logger.info(f"Scorer loaded — threshold={self.threshold:.3f}, ROC-AUC={self.roc_auc:.3f}")

    def score(self, amount: float, hour: int, day_of_week: int,
              merchant_category: str, velocity: dict) -> dict:
        """
        Assemble feature vector → scale → predict → return prob + decision + signals.
        velocity dict comes from VelocityEngine.get_velocity_features().
        """

        # One-hot encode merchant category against known categories
        cat_vec = [int(merchant_category == c) for c in CATEGORIES]

        # Build feature vector in exact same order as FEATURES list in train.py
        feature_vec = np.array([[amount, hour, day_of_week] + cat_vec], dtype=float)
        feature_vec_scaled = self.scaler.transform(feature_vec)

        # Model inference — get fraud probability from class-1 column
        prob = float(self.model.predict_proba(feature_vec_scaled)[0][1])
        decision = "BLOCK" if prob >= self.threshold else "ALLOW"

        # Build top-3 signals for explainability in the response
        signals = self._build_signals(amount, velocity, prob)

        return {
            "fraud_probability": round(prob, 4),
            "decision": decision,
            "threshold_used": round(self.threshold, 3),
            "top_signals": signals,
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
