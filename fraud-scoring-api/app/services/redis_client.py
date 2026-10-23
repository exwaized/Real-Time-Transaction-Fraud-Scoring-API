"""
Shared Redis client for state that must be consistent across workers/replicas
(velocity windows, behavioral profiles, rate limits).

REDIS_URL unset or unreachable at startup -> get_redis() returns None and each
service keeps its in-process fallback, same switch as divergence.py.
"""
import os
import time
import logging

logger = logging.getLogger("redis_client")

RETRY_INTERVAL_SEC = 15

_client = None
_url_checked = False
_next_retry = 0.0


def _connect():
    import redis
    client = redis.from_url(
        os.environ["REDIS_URL"], socket_connect_timeout=2, socket_timeout=1, decode_responses=True
    )
    client.ping()
    return client


def get_redis():
    """
    Return the shared client, or None to use in-process state.

    If REDIS_URL is set but Redis was unreachable, retry at most every
    RETRY_INTERVAL_SEC so the service recovers shared state on its own once
    Redis comes back instead of staying per-process until restart.
    """
    global _client, _url_checked, _next_retry
    if _client is not None:
        return _client
    if not os.getenv("REDIS_URL"):
        if not _url_checked:
            _url_checked = True
            logger.info("REDIS_URL not set — velocity/profile/rate-limit state is per-process")
        return None

    now = time.monotonic()
    if now < _next_retry:
        return None
    _next_retry = now + RETRY_INTERVAL_SEC
    try:
        _client = _connect()
        logger.info("Shared state backed by Redis — consistent across workers")
    except Exception as e:
        logger.warning(
            f"REDIS_URL set but Redis unreachable ({e}); using in-process state, "
            f"retrying in {RETRY_INTERVAL_SEC}s"
        )
    return _client


def set_redis_for_tests(client) -> None:
    """Inject a client (e.g. fakeredis) or None to force the in-memory fallback."""
    global _client, _url_checked, _next_retry
    _client = client
    _url_checked = True
    _next_retry = float("inf") if client is None else 0.0
