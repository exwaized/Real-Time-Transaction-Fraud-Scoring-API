"""
Sliding window rate limiter.
Per-key: max N requests per window_sec seconds.

Redis-backed (sorted set per key) when REDIS_URL is set, so the limit holds
across all workers/replicas; otherwise an in-process deque guarded by an RLock.
If Redis errors mid-flight the check falls back to the in-process window rather
than failing the request.
"""
import logging
import time
import threading
import uuid
from collections import defaultdict, deque
from fastapi import HTTPException, status

from app.services.redis_client import get_redis

logger = logging.getLogger("rate_limit")


class RateLimiter:
    def __init__(self, max_requests: int = 60, window_sec: int = 60):
        # 60 req/min per API key by default
        self.max_requests = max_requests
        self.window_sec   = window_sec
        self._lock        = threading.RLock()
        # key → deque of timestamps (fallback store)
        self._windows: dict[str, deque] = defaultdict(deque)

    def check(self, key: str) -> None:
        """Raise 429 if key has exceeded the rate limit."""
        r = get_redis()
        if r is not None:
            try:
                allowed = self._check_redis(r, key)
            except Exception as e:
                logger.warning(f"Redis rate-limit check failed ({e}); using in-process window")
            else:
                if not allowed:
                    self._raise_429()
                return
        if not self._check_local(key):
            self._raise_429()

    def _check_redis(self, r, key: str) -> bool:
        now = time.time()
        rkey = f"fraud:rl:{key}"
        member = f"{now}:{uuid.uuid4().hex}"
        # Add first, then count: concurrent callers can't both slip under the cap.
        pipe = r.pipeline(transaction=True)
        pipe.zadd(rkey, {member: now})
        pipe.zremrangebyscore(rkey, 0, now - self.window_sec)
        pipe.zcard(rkey)
        pipe.expire(rkey, self.window_sec + 1)
        _, _, count, _ = pipe.execute()
        if count > self.max_requests:
            r.zrem(rkey, member)   # rejected requests don't consume quota
            return False
        return True

    def _check_local(self, key: str) -> bool:
        now     = time.time()
        cutoff  = now - self.window_sec
        with self._lock:
            window = self._windows[key]
            while window and window[0] < cutoff:
                window.popleft()
            if len(window) >= self.max_requests:
                return False
            window.append(now)
            return True

    def _raise_429(self):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {self.max_requests} req/{self.window_sec}s",
            headers={"Retry-After": str(self.window_sec)},
        )


# Singleton
rate_limiter = RateLimiter(max_requests=60, window_sec=60)
