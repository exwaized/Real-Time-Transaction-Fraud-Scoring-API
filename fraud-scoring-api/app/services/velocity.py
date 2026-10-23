"""
Velocity feature engine.
Maintains per-card rolling windows and behavioral profiles.

Redis-backed when REDIS_URL is set (per-card list + profile hash, both with a
TTL), so every worker/replica sees the same card history. Otherwise — or if a
Redis call fails — falls back to the in-process deque + RLock store.
"""
import json
import logging
import threading
import time
import datetime
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List

from app.services import redis_client
from app.services.redis_client import get_redis
from app.services.privacy import card_ref

logger = logging.getLogger("velocity")

WINDOW_MAXLEN = 500            # ≈ 7d proxy
STATE_TTL_SEC = 7 * 86400      # idle cards age out of Redis


@dataclass
class TxnRecord:
    timestamp: float
    amount: float
    merchant_id: str
    merchant_category: str


class VelocityEngine:
    def __init__(self):
        self._lock = threading.RLock()
        # card_id → deque of TxnRecord (fallback store)
        self._store: Dict[str, deque] = defaultdict(lambda: deque(maxlen=WINDOW_MAXLEN))
        self._profile_lock = threading.RLock()
        self._profiles: Dict[str, dict] = {}   # fallback behavioral profiles

    # ── writes ────────────────────────────────────────────────────────────
    def record(self, card_id: str, amount: float, merchant_id: str, category: str):
        """Append a new transaction to the card's rolling window."""
        rec = TxnRecord(time.time(), amount, merchant_id, category)
        r = get_redis()
        if r is not None:
            try:
                self._record_redis(r, card_id, rec)
                redis_client.mark_ok()
                return
            except Exception as e:
                redis_client.mark_failure(e)
                logger.warning(f"Redis velocity write failed ({e}); recording in-process")
        with self._lock:
            self._store[card_id].append(rec)
        self._update_profile(card_id, rec)

    def _record_redis(self, r, card_id: str, rec: TxnRecord):
        ref = card_ref(card_id)
        wkey, pkey = f"fraud:vel:{ref}", f"fraud:prof:{ref}"
        hour = datetime.datetime.fromtimestamp(rec.timestamp).hour
        pipe = r.pipeline(transaction=True)
        pipe.rpush(wkey, json.dumps(asdict(rec)))
        pipe.ltrim(wkey, -WINDOW_MAXLEN, -1)
        pipe.expire(wkey, STATE_TTL_SEC)
        pipe.hincrby(pkey, "txn_count", 1)
        pipe.hincrbyfloat(pkey, "amount_sum", rec.amount)
        pipe.hincrby(pkey, f"hour:{hour}", 1)
        pipe.hincrby(pkey, f"cat:{rec.merchant_category}", 1)
        pipe.expire(pkey, STATE_TTL_SEC)
        pipe.execute()

    # ── reads ─────────────────────────────────────────────────────────────
    def _load_records(self, card_id: str) -> List[TxnRecord]:
        r = get_redis()
        if r is not None:
            try:
                recs = [TxnRecord(**json.loads(x))
                        for x in r.lrange(f"fraud:vel:{card_ref(card_id)}", 0, -1)]
                redis_client.mark_ok()
                return recs
            except Exception as e:
                redis_client.mark_failure(e)
                logger.warning(f"Redis velocity read failed ({e}); using in-process window")
        with self._lock:
            return list(self._store.get(card_id, []))

    def get_velocity_features(self, card_id: str, amount: float) -> dict:
        """Compute velocity features for scoring. O(n) over window — kept small."""
        now = time.time()
        window_1hr  = now - 3600
        window_24hr = now - 86400

        records = self._load_records(card_id)

        txns_1hr  = [r for r in records if r.timestamp >= window_1hr]
        txns_24hr = [r for r in records if r.timestamp >= window_24hr]

        amounts_7d = [r.amount for r in records]
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
        """Lightweight incremental behavioral profile update (in-process fallback)."""
        with self._profile_lock:
            p = self._profiles.setdefault(card_id, {
                "txn_count": 0, "amount_sum": 0.0,
                "hour_counts": defaultdict(int),
                "categories": defaultdict(int),
            })
            p["txn_count"] += 1
            p["amount_sum"] += record.amount
            hour = datetime.datetime.fromtimestamp(record.timestamp).hour
            p["hour_counts"][str(hour)] += 1
            p["categories"][record.merchant_category] += 1

    def _load_profile(self, card_id: str):
        """Return {txn_count, amount_sum, hour_counts, categories} or None."""
        r = get_redis()
        if r is not None:
            try:
                h = r.hgetall(f"fraud:prof:{card_ref(card_id)}")
                if not h:
                    return None
                return {
                    "txn_count": int(h["txn_count"]),
                    "amount_sum": float(h["amount_sum"]),
                    "hour_counts": {k[5:]: int(v) for k, v in h.items() if k.startswith("hour:")},
                    "categories": {k[4:]: int(v) for k, v in h.items() if k.startswith("cat:")},
                }
            except Exception as e:
                redis_client.mark_failure(e)
                logger.warning(f"Redis profile read failed ({e}); using in-process profile")
        with self._profile_lock:
            p = self._profiles.get(card_id)
            if not p:
                return None
            return {
                "txn_count": p["txn_count"], "amount_sum": p["amount_sum"],
                "hour_counts": dict(p["hour_counts"]), "categories": dict(p["categories"]),
            }

    def get_profile(self, card_id: str) -> dict:
        p = self._load_profile(card_id)
        if not p:
            return {}
        avg_amount = p["amount_sum"] / max(p["txn_count"], 1)
        top_hour = max(p["hour_counts"], key=p["hour_counts"].get, default=None)
        top_cat  = max(p["categories"],  key=p["categories"].get,  default=None)
        return {
            "avg_amount": round(avg_amount, 2),
            "top_hour": int(top_hour) if top_hour is not None else None,
            "top_category": top_cat,
            "total_txns_seen": p["txn_count"],
        }


# Singleton — shared across request workers
velocity_engine = VelocityEngine()
