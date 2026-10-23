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

logger = logging.getLogger("divergence")

REDIS_KEY = "fraud:divergence"


class DivergenceLog:
    def __init__(self, maxlen: int = 100):
        self.maxlen = maxlen
        self._records = deque(maxlen=maxlen)   # fallback store
        self._lock = threading.Lock()
        self._redis = None

        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            logger.info("REDIS_URL not set — divergence log is per-process (single-worker only)")
            return

        try:
            import redis
            client = redis.from_url(redis_url, socket_connect_timeout=2, decode_responses=True)
            client.ping()
            self._redis = client
            logger.info(f"DivergenceLog backed by Redis ({redis_url}) — shared across workers")
        except Exception as e:
            logger.warning(f"REDIS_URL set but Redis unreachable ({e}); falling back to in-process log")

    def record(self, champion_prob: float, challenger_prob: float, routed_model: str) -> None:
        entry = {
            "ts": round(time.time(), 3),
            "champion_score": round(champion_prob, 4),
            "challenger_score": round(challenger_prob, 4),
            "delta": round(abs(champion_prob - challenger_prob), 4),
            "routed_model": routed_model,
        }

        if self._redis is not None:
            try:
                pipe = self._redis.pipeline()
                pipe.rpush(REDIS_KEY, json.dumps(entry))
                pipe.ltrim(REDIS_KEY, -self.maxlen, -1)
                pipe.execute()
                return
            except Exception as e:
                logger.warning(f"Redis write failed ({e}); recording in-process instead")

        with self._lock:
            self._records.append(entry)

    def get_records(self) -> list:
        if self._redis is not None:
            try:
                raw = self._redis.lrange(REDIS_KEY, 0, -1)
                return [json.loads(r) for r in raw]
            except Exception as e:
                logger.warning(f"Redis read failed ({e}); serving in-process log")

        with self._lock:
            return list(self._records)


# Singleton
divergence_log = DivergenceLog()
