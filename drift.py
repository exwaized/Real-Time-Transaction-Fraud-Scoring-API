"""
Drift Detection Monitor — background daemon thread.
Tracks score distribution + feature PSI, fires alert when drift detected.
"""
import threading
import time
import logging
import json
from collections import deque
from typing import Optional

logger = logging.getLogger("drift_monitor")

class DriftMonitor:
    """
    Runs as a daemon thread. Every `check_interval` seconds it computes:
    - Score distribution drift (mean shift)
    - PSI on txn_count_1hr and amount_zscore
    Fires Slack webhook or logs CRITICAL alert on breach.
    """

    SCORE_DRIFT_THRESHOLD  = 0.15   # absolute mean shift
    PSI_THRESHOLD          = 0.20   # standard PSI alert level
    CHECK_INTERVAL_SEC     = 60

    def __init__(self, slack_webhook: Optional[str] = None):
        self._scores: deque  = deque(maxlen=5000)
        self._features: deque = deque(maxlen=5000)  # dicts
        self._baseline_score_mean: Optional[float] = None
        self._baseline_features: Optional[dict] = None
        self._lock = threading.Lock()
        self._slack_webhook = slack_webhook
        self._alerts: list = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("DriftMonitor started")

    def record(self, score: float, features: dict):
        with self._lock:
            self._scores.append(score)
            self._features.append(features)
            # First 200 records become baseline
            if len(self._scores) == 200 and self._baseline_score_mean is None:
                self._baseline_score_mean = sum(list(self._scores)) / 200
                logger.info(f"Baseline score mean set: {self._baseline_score_mean:.4f}")

    def _run(self):
        while True:
            time.sleep(self.CHECK_INTERVAL_SEC)
            try:
                self._check()
            except Exception as e:
                logger.error(f"Drift check error: {e}")

    def _check(self):
        with self._lock:
            scores = list(self._scores)
            feats  = list(self._features)

        if len(scores) < 50 or self._baseline_score_mean is None:
            return

        recent_mean = sum(scores[-100:]) / min(len(scores), 100)
        score_drift = abs(recent_mean - self._baseline_score_mean)

        psi_result = self._compute_psi(
            [f.get("amount_zscore", 0) for f in feats[:200]],
            [f.get("amount_zscore", 0) for f in feats[-100:]],
        )

        alerts = []
        if score_drift > self.SCORE_DRIFT_THRESHOLD:
            alerts.append(f"SCORE_DRIFT: baseline={self._baseline_score_mean:.3f} recent={recent_mean:.3f} Δ={score_drift:.3f}")
        if psi_result > self.PSI_THRESHOLD:
            alerts.append(f"FEATURE_PSI: amount_zscore PSI={psi_result:.3f} > {self.PSI_THRESHOLD}")

        for alert in alerts:
            logger.critical(f"[DRIFT ALERT] {alert}")
            self._alerts.append({"ts": time.time(), "message": alert})
            if self._slack_webhook:
                self._fire_slack(alert)

    def _compute_psi(self, baseline: list, current: list, bins: int = 10) -> float:
        """Population Stability Index."""
        import numpy as np
        if not baseline or not current:
            return 0.0
        b = np.array(baseline); c = np.array(current)
        all_vals = np.concatenate([b, c])
        edges = np.percentile(all_vals, np.linspace(0, 100, bins + 1))
        edges = np.unique(edges)
        if len(edges) < 2:
            return 0.0
        b_counts = np.histogram(b, bins=edges)[0] / max(len(b), 1)
        c_counts = np.histogram(c, bins=edges)[0] / max(len(c), 1)
        b_counts = np.where(b_counts == 0, 1e-6, b_counts)
        c_counts = np.where(c_counts == 0, 1e-6, c_counts)
        return float(np.sum((c_counts - b_counts) * np.log(c_counts / b_counts)))

    def _fire_slack(self, message: str):
        try:
            import urllib.request
            payload = json.dumps({"text": f":rotating_light: *Fraud API Drift Alert*\n{message}"}).encode()
            req = urllib.request.Request(
                self._slack_webhook,
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            urllib.request.urlopen(req, timeout=3)
        except Exception as e:
            logger.warning(f"Slack webhook failed: {e}")

    def get_status(self) -> dict:
        with self._lock:
            scores = list(self._scores)
        return {
            "total_scored": len(scores),
            "recent_mean_score": round(sum(scores[-100:]) / max(len(scores[-100:]), 1), 4) if scores else None,
            "baseline_mean_score": round(self._baseline_score_mean, 4) if self._baseline_score_mean else None,
            "recent_alerts": self._alerts[-5:],
        }

# Singleton
drift_monitor = DriftMonitor()
