"""
Champion/Challenger Divergence Log.
Every scored request runs both models; this records how far the challenger's
score strays from the champion's, for offline comparison via /challenger/divergence.

Backed by Redis when REDIS_URL is set, so the divergence feed is one shared list
across every worker/replica — the point being demonstrated is that a card scored
by worker A and one scored by worker B land in the same feed. Falls back to an
in-process deque when REDIS_URL is unset (same "one env var is the switch"
pattern as FRAUD_API_KEY in app/services/auth.py), so local dev needs no Redis.
"""
import os
import json
import time
import logging
import threading
from collections import deque

from app.services import redis_client
from app.services.redis_client import get_redis

logger = logging.getLogger("divergence")

REDIS_KEY = "fraud:divergence"


class DivergenceLog:
    def __init__(self, maxlen: int = 100):
        self.maxlen = maxlen
        self._records = deque(maxlen=maxlen)   # fallback store
        self._lock = threading.Lock()

    def record(self, champion_prob: float, challenger_prob: float, routed_model: str) -> None:
        entry = {
            "ts": round(time.time(), 3),
            "champion_score": round(champion_prob, 4),
            "challenger_score": round(challenger_prob, 4),
            "delta": round(abs(champion_prob - challenger_prob), 4),
            "routed_model": routed_model,
        }

        r = get_redis()
        if r is not None:
            try:
                pipe = r.pipeline()
                pipe.rpush(REDIS_KEY, json.dumps(entry))
                pipe.ltrim(REDIS_KEY, -self.maxlen, -1)
                pipe.execute()
                redis_client.mark_ok()
                return
            except Exception as e:
                redis_client.mark_failure(e)
                logger.warning(f"Redis write failed ({e}); recording in-process instead")

        with self._lock:
            self._records.append(entry)

    def get_records(self) -> list:
        r = get_redis()
        if r is not None:
            try:
                raw = r.lrange(REDIS_KEY, 0, -1)
                redis_client.mark_ok()
                return [json.loads(x) for x in raw]
            except Exception as e:
                redis_client.mark_failure(e)
                logger.warning(f"Redis read failed ({e}); serving in-process log")

        with self._lock:
            return list(self._records)


# Singleton
divergence_log = DivergenceLog()
