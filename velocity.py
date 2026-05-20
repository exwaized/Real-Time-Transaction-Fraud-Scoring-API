"""
Thread-safe velocity feature engine.
Maintains per-card rolling windows using deque + RLock (reused from anomaly engine pattern).
"""
import threading
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class TxnRecord:
    timestamp: float
    amount: float
    merchant_id: str
    merchant_category: str

class VelocityEngine:
    """
    Rolling window store for per-card velocity features.
    Thread-safe via RLock + double-checked locking pattern.
    """

    def __init__(self):
        self._lock = threading.RLock()
        # card_id → deque of TxnRecord (kept sorted by timestamp)
        self._store: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self._profile_lock = threading.RLock()
        self._profiles: Dict[str, dict] = {}   # behavioral profiles

    def record(self, card_id: str, amount: float, merchant_id: str, category: str):
        """Append a new transaction to the card's rolling window."""
        record = TxnRecord(
            timestamp=time.time(),
            amount=amount,
            merchant_id=merchant_id,
            merchant_category=category
        )
        with self._lock:
            self._store[card_id].append(record)
        self._update_profile(card_id, record)

    def get_velocity_features(self, card_id: str, amount: float) -> dict:
        """Compute velocity features for scoring. O(n) over window — kept small."""
        now = time.time()
        window_1hr  = now - 3600
        window_24hr = now - 86400

        with self._lock:
            records: List[TxnRecord] = list(self._store.get(card_id, []))

        txns_1hr  = [r for r in records if r.timestamp >= window_1hr]
        txns_24hr = [r for r in records if r.timestamp >= window_24hr]

        amounts_7d = [r.amount for r in records]  # maxlen=500 ≈ 7d proxy
        mean_7d = float(sum(amounts_7d) / len(amounts_7d)) if amounts_7d else amount
        std_7d  = float((sum((a - mean_7d)**2 for a in amounts_7d) / max(len(amounts_7d),1))**0.5)
        zscore  = (amount - mean_7d) / (std_7d + 1e-9)

        merchants_1hr = {r.merchant_id for r in txns_1hr}
        last_ts = records[-1].timestamp if records else None
        time_since_last = (now - last_ts) if last_ts else None

        return {
            "txn_count_1hr": len(txns_1hr),
            "txn_count_24hr": len(txns_24hr),
            "amount_mean_7d": round(mean_7d, 2),
            "amount_zscore": round(zscore, 3),
            "unique_merchants_1hr": len(merchants_1hr),
            "time_since_last_txn_sec": round(time_since_last, 1) if time_since_last else None,
        }

    def _update_profile(self, card_id: str, record: TxnRecord):
        """Lightweight incremental behavioral profile update."""
        with self._profile_lock:
            p = self._profiles.setdefault(card_id, {
                "txn_count": 0, "amount_sum": 0.0,
                "hour_counts": defaultdict(int),
                "categories": defaultdict(int),
            })
            p["txn_count"] += 1
            p["amount_sum"] += record.amount
            import datetime
            hour = datetime.datetime.fromtimestamp(record.timestamp).hour
            p["hour_counts"][str(hour)] += 1
            p["categories"][record.merchant_category] += 1

    def get_profile(self, card_id: str) -> dict:
        with self._profile_lock:
            p = self._profiles.get(card_id)
            if not p:
                return {}
            avg_amount = p["amount_sum"] / max(p["txn_count"], 1)
            top_hour = max(p["hour_counts"], key=p["hour_counts"].get, default=None)
            top_cat  = max(p["categories"],  key=p["categories"].get,  default=None)
            return {
                "avg_amount": round(avg_amount, 2),
                "top_hour": int(top_hour) if top_hour else None,
                "top_category": top_cat,
                "total_txns_seen": p["txn_count"],
            }

# Singleton — shared across request workers
velocity_engine = VelocityEngine()
