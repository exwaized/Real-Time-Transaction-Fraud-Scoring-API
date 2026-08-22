"""
In-memory sliding window rate limiter.
Per-key: max N requests per window_sec seconds.
Thread-safe via RLock (same pattern as VelocityEngine).
"""
import time
import threading
from collections import defaultdict, deque
from fastapi import HTTPException, status

class RateLimiter:
    def __init__(self, max_requests: int = 60, window_sec: int = 60):
        # 60 req/min per API key by default
        self.max_requests = max_requests
        self.window_sec   = window_sec
        self._lock        = threading.RLock()
        # key → deque of timestamps
        self._windows: dict[str, deque] = defaultdict(deque)

    def check(self, key: str) -> None:
        """Raise 429 if key has exceeded the rate limit."""
        now     = time.time()
        cutoff  = now - self.window_sec

        with self._lock:
            window = self._windows[key]

            # Drop timestamps outside the current window
            while window and window[0] < cutoff:
                window.popleft()

            if len(window) >= self.max_requests:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded: {self.max_requests} req/{self.window_sec}s",
                    headers={"Retry-After": str(self.window_sec)},
                )

            window.append(now)

# Singleton — shared across workers
rate_limiter = RateLimiter(max_requests=60, window_sec=60)
