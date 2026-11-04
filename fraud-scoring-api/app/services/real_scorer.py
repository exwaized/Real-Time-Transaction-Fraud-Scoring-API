"""
Scorer for the model trained on real data (train_real.py).

The ULB dataset exposes 28 anonymised PCA components (V1..V28) plus the
transaction amount, so this scorer takes those directly instead of the
card/merchant fields used by the demo /score endpoint.
"""
import json
import pickle
import time
from pathlib import Path

import numpy as np

_DIR = Path(__file__).parent.parent / "models"


class RealScorer:
    def __init__(self):
        self.model = None
        self.meta = {}
        model_path, meta_path = _DIR / "model_real.pkl", _DIR / "meta_real.json"
        if model_path.exists() and meta_path.exists():
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            with open(meta_path) as f:
                self.meta = json.load(f)

    @property
    def available(self) -> bool:
        return self.model is not None

    def score(self, amount: float, v: list[float]) -> dict:
        feats = self.meta["features"]
        row = np.array([[np.log1p(amount)] + list(v)], dtype=float)
        assert row.shape[1] == len(feats)
        t0 = time.perf_counter()
        prob = float(self.model.predict_proba(row)[0][1])
        latency = round((time.perf_counter() - t0) * 1000, 3)
        thr = self.meta["threshold"]
        return {
            "fraud_probability": round(prob, 4),
            "decision": "BLOCK" if prob >= thr else "ALLOW",
            "threshold_used": round(thr, 4),
            "latency_ms": latency,
        }


real_scorer = RealScorer()
