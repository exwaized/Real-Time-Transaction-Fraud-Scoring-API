"""
Shared-state tests: velocity, profiles and rate limits must be consistent across
"workers" (separate engine instances) when Redis is configured, and degrade to
in-process state when it isn't or when it fails.
"""
import sys
from pathlib import Path

import fakeredis
import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services import redis_client  # noqa: E402
from app.services.rate_limit import RateLimiter  # noqa: E402
from app.services.velocity import VelocityEngine  # noqa: E402
from app.services.privacy import card_ref  # noqa: E402


@pytest.fixture
def fake_redis():
    client = fakeredis.FakeRedis(decode_responses=True)
    redis_client.set_redis_for_tests(client)
    yield client
    redis_client.set_redis_for_tests(None)


@pytest.fixture
def no_redis():
    redis_client.set_redis_for_tests(None)
    yield


class BrokenRedis:
    def __getattr__(self, name):
        def fail(*a, **k):
            raise ConnectionError("redis down")
        return fail


class TestVelocityShared:
    def test_history_visible_across_workers(self, fake_redis):
        worker_a, worker_b = VelocityEngine(), VelocityEngine()
        for _ in range(3):
            worker_a.record("c1", 100.0, "m1", "grocery")
        feats = worker_b.get_velocity_features("c1", 100.0)
        assert feats["txn_count_1hr"] == 3
        assert feats["amount_mean_7d"] == 100.0

    def test_profile_visible_across_workers(self, fake_redis):
        worker_a, worker_b = VelocityEngine(), VelocityEngine()
        worker_a.record("c2", 50.0, "m1", "fuel")
        worker_a.record("c2", 150.0, "m2", "fuel")
        p = worker_b.get_profile("c2")
        assert p["total_txns_seen"] == 2
        assert p["avg_amount"] == 100.0
        assert p["top_category"] == "fuel"
        assert isinstance(p["top_hour"], int)

    def test_unknown_card_empty_profile(self, fake_redis):
        assert VelocityEngine().get_profile("nobody") == {}

    def test_window_capped_and_ttl_set(self, fake_redis):
        eng = VelocityEngine()
        for _ in range(510):
            eng.record("c3", 10.0, "m", "online")
        assert fake_redis.llen(f"fraud:vel:{card_ref('c3')}") == 500
        assert 0 < fake_redis.ttl(f"fraud:vel:{card_ref('c3')}") <= 7 * 86400
        assert 0 < fake_redis.ttl(f"fraud:prof:{card_ref('c3')}") <= 7 * 86400

    def test_redis_failure_falls_back_in_process(self):
        redis_client.set_redis_for_tests(BrokenRedis())
        try:
            eng = VelocityEngine()
            eng.record("c4", 20.0, "m", "atm")
            assert eng.get_velocity_features("c4", 20.0)["txn_count_1hr"] == 1
            assert eng.get_profile("c4")["total_txns_seen"] == 1
        finally:
            redis_client.set_redis_for_tests(None)

    def test_no_redis_uses_memory(self, no_redis):
        a, b = VelocityEngine(), VelocityEngine()
        a.record("c5", 20.0, "m", "atm")
        assert b.get_velocity_features("c5", 20.0)["txn_count_1hr"] == 0  # per-process


class TestRateLimitShared:
    def test_cap_shared_across_workers(self, fake_redis):
        a = RateLimiter(max_requests=5, window_sec=60)
        b = RateLimiter(max_requests=5, window_sec=60)
        for i in range(5):
            (a if i % 2 else b).check("k")
        with pytest.raises(HTTPException) as exc:
            a.check("k")
        assert exc.value.status_code == 429
        with pytest.raises(HTTPException):
            b.check("k")

    def test_rejected_requests_do_not_consume_quota(self, fake_redis):
        rl = RateLimiter(max_requests=2, window_sec=60)
        rl.check("k"); rl.check("k")
        for _ in range(5):
            with pytest.raises(HTTPException):
                rl.check("k")
        assert fake_redis.zcard("fraud:rl:k") == 2

    def test_keys_are_independent(self, fake_redis):
        rl = RateLimiter(max_requests=1, window_sec=60)
        rl.check("a")
        rl.check("b")

    def test_redis_failure_falls_back_in_process(self):
        redis_client.set_redis_for_tests(BrokenRedis())
        try:
            rl = RateLimiter(max_requests=1, window_sec=60)
            rl.check("k")
            with pytest.raises(HTTPException):
                rl.check("k")
        finally:
            redis_client.set_redis_for_tests(None)

    def test_no_redis_uses_memory(self, no_redis):
        rl = RateLimiter(max_requests=1, window_sec=60)
        rl.check("k")
        with pytest.raises(HTTPException):
            rl.check("k")


class TestRedisReconnect:
    def _reset(self, monkeypatch):
        monkeypatch.setenv("REDIS_URL", "redis://unreachable:6379/0")
        monkeypatch.setattr(redis_client, "_client", None)
        monkeypatch.setattr(redis_client, "_next_retry", 0.0)

    def test_starts_degraded_then_recovers(self, monkeypatch):
        self._reset(monkeypatch)
        calls = {"n": 0}
        good = fakeredis.FakeRedis(decode_responses=True)

        def flaky():
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionError("down")
            return good

        monkeypatch.setattr(redis_client, "_connect", flaky)
        clock = {"t": 1000.0}
        monkeypatch.setattr(redis_client.time, "monotonic", lambda: clock["t"])

        assert redis_client.get_redis() is None          # down at first use
        assert redis_client.get_redis() is None          # inside retry window: no new attempt
        assert calls["n"] == 1
        clock["t"] += redis_client.RETRY_INTERVAL_SEC + 1
        assert redis_client.get_redis() is good          # Redis is back: reconnects
        assert redis_client.get_redis() is good          # and stays connected, no more attempts
        assert calls["n"] == 2

    def test_unreachable_does_not_hammer(self, monkeypatch):
        self._reset(monkeypatch)
        calls = {"n": 0}

        def down():
            calls["n"] += 1
            raise ConnectionError("down")

        monkeypatch.setattr(redis_client, "_connect", down)
        for _ in range(100):
            assert redis_client.get_redis() is None
        assert calls["n"] == 1


class TestRedisDegradedSignal:
    def test_failure_flips_gauge_and_skips_redis_until_retry(self, monkeypatch):
        from app.services.metrics import SHARED_STATE_DEGRADED
        redis_client.set_redis_for_tests(BrokenRedis())
        try:
            rl = RateLimiter(max_requests=5, window_sec=60)
            rl.check("k")                              # Redis call fails -> degraded
            assert SHARED_STATE_DEGRADED._value.get() == 1
            assert redis_client.get_redis() is None    # later calls skip Redis (no timeout tax)
        finally:
            redis_client.set_redis_for_tests(None)

    def test_success_after_failure_restores_gauge(self, fake_redis):
        from app.services.metrics import SHARED_STATE_DEGRADED
        redis_client.mark_failure(ConnectionError("x"))
        assert SHARED_STATE_DEGRADED._value.get() == 1
        redis_client.set_redis_for_tests(fake_redis)   # Redis is back
        redis_client._degraded = True
        RateLimiter(max_requests=5, window_sec=60).check("k")
        assert SHARED_STATE_DEGRADED._value.get() == 0


class TestPrivacy:
    def test_card_ids_not_in_redis_keys(self, fake_redis):
        VelocityEngine().record("4111111111111111", 10.0, "m", "online")
        keys = " ".join(fake_redis.keys("*"))
        assert "4111111111111111" not in keys
        assert card_ref("4111111111111111") in keys

    def test_salt_changes_ref(self, monkeypatch):
        a = card_ref("c")
        monkeypatch.setenv("CARD_HASH_SALT", "pepper")
        assert card_ref("c") != a
