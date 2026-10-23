"""
Tests for Fraud Scoring API.
Run: pytest tests/ -v

Requires trained artifacts in app/models/ (run train.py first); skipped otherwise.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

MODELS = Path(__file__).parent.parent / "app" / "models"
if not (MODELS / "champion_model.pkl").exists():
    pytest.skip("trained model artifacts missing", allow_module_level=True)

TEST_KEY = "test-key-123"
os.environ["FRAUD_API_KEY"] = TEST_KEY
os.environ.pop("REDIS_URL", None)

from fastapi.testclient import TestClient  # noqa: E402
from main import app  # noqa: E402
from app.services.rate_limit import RateLimiter  # noqa: E402
from app.services import rate_limit as rl_module  # noqa: E402
from app.services.adjudicator import Adjudicator  # noqa: E402

AUTH = {"X-API-Key": TEST_KEY}

BASE_TXN = {
    "card_id": "card_0042",
    "merchant_id": "merch_017",
    "amount": 450.0,
    "merchant_category": "electronics",
    "timestamp": "2025-05-19T14:22:11",
}


@pytest.fixture(autouse=True)
def fresh_rate_limiter(monkeypatch):
    """Each test gets an empty limiter so the 60/min cap doesn't leak across tests."""
    limiter = RateLimiter(max_requests=60, window_sec=60)
    monkeypatch.setattr(rl_module, "rate_limiter", limiter)
    import app.routers.score as score_module
    monkeypatch.setattr(score_module, "rate_limiter", limiter)
    return limiter


@pytest.fixture
def client():
    return TestClient(app)


class TestAuth:
    def test_missing_key_rejected(self, client):
        assert client.post("/score", json=BASE_TXN).status_code == 401

    def test_wrong_key_rejected(self, client):
        r = client.post("/score", json=BASE_TXN, headers={"X-API-Key": "nope"})
        assert r.status_code == 401

    def test_unset_key_fails_closed(self, client, monkeypatch):
        monkeypatch.delenv("FRAUD_API_KEY")
        monkeypatch.delenv("FRAUD_API_KEYS", raising=False)
        monkeypatch.delenv("ALLOW_INSECURE_NO_AUTH", raising=False)
        assert client.post("/score", json=BASE_TXN).status_code == 503
        assert client.post("/score", json=BASE_TXN, headers=AUTH).status_code == 503

    def test_explicit_insecure_opt_in(self, client, monkeypatch):
        monkeypatch.delenv("FRAUD_API_KEY")
        monkeypatch.setenv("ALLOW_INSECURE_NO_AUTH", "true")
        assert client.post("/score", json=BASE_TXN).status_code == 200


class TestPerClientKeys:
    def test_each_client_key_works_and_unknown_rejected(self, client, monkeypatch):
        monkeypatch.setenv("FRAUD_API_KEYS", "payments=key-pay,risk=key-risk")
        for k in ("key-pay", "key-risk", TEST_KEY):
            assert client.get("/health", headers={"X-API-Key": k}).status_code == 200
        assert client.get("/health", headers={"X-API-Key": "key-other"}).status_code == 401

    def test_limits_are_per_client(self, client, monkeypatch):
        monkeypatch.setenv("FRAUD_API_KEYS", "payments=key-pay,risk=key-risk")
        monkeypatch.setenv("FRAUD_RATE_LIMITS", "payments=5,risk=2")
        pay = [client.get("/health", headers={"X-API-Key": "key-pay"}).status_code for _ in range(6)]
        risk = [client.get("/health", headers={"X-API-Key": "key-risk"}).status_code for _ in range(3)]
        assert pay == [200] * 5 + [429]
        assert risk == [200, 200, 429]      # one client's quota doesn't affect another

    def test_secret_never_used_as_rate_limit_key(self, client, monkeypatch, fresh_rate_limiter):
        monkeypatch.setenv("FRAUD_API_KEYS", "payments=key-pay-secret")
        client.get("/health", headers={"X-API-Key": "key-pay-secret"})
        assert "key-pay-secret" not in " ".join(fresh_rate_limiter._windows)
        assert "payments" in fresh_rate_limiter._windows

    def test_invalid_limit_value_falls_back_to_default(self, client, monkeypatch):
        monkeypatch.setenv("FRAUD_RATE_LIMITS", "default=notanumber")
        assert client.get("/health", headers=AUTH).status_code == 200


class TestLogPrivacy:
    def test_card_id_not_logged_in_plaintext(self, client, caplog):
        import logging
        with caplog.at_level(logging.INFO, logger="router"):
            client.post("/score", json={**BASE_TXN, "card_id": "4111111111111111"}, headers=AUTH)
        assert "SCORED" in caplog.text
        assert "4111111111111111" not in caplog.text


class TestHealth:
    def test_healthz_is_unauthenticated(self, client):
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_health_requires_key(self, client):
        assert client.get("/health").status_code == 401

    def test_health_ok(self, client):
        r = client.get("/health", headers=AUTH)
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_healthz_does_not_consume_rate_limit(self, client, fresh_rate_limiter):
        for _ in range(100):
            assert client.get("/healthz").status_code == 200

    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "docs" in r.json()


class TestScoreEndpoint:
    def test_score_schema(self, client):
        r = client.post("/score", json=BASE_TXN, headers=AUTH)
        assert r.status_code == 200
        body = r.json()
        for field in ("card_id", "fraud_probability", "champion_probability",
                      "challenger_probability", "model_used", "decision",
                      "threshold_used", "top_signals", "velocity",
                      "behavioral_profile", "latency_ms", "reviewed"):
            assert field in body

    def test_probability_in_range_and_decision_valid(self, client):
        body = client.post("/score", json=BASE_TXN, headers=AUTH).json()
        assert 0.0 <= body["fraud_probability"] <= 1.0
        assert body["decision"] in ("ALLOW", "BLOCK", "REVIEW")
        assert body["model_used"] in ("champion_lgbm", "challenger_xgb")

    def test_top_signals_at_most_three(self, client):
        body = client.post("/score", json=BASE_TXN, headers=AUTH).json()
        assert 1 <= len(body["top_signals"]) <= 3

    def test_invalid_amount_rejected(self, client):
        r = client.post("/score", json={**BASE_TXN, "amount": -5}, headers=AUTH)
        assert r.status_code == 422

    def test_bad_timestamp_is_422_not_500(self, client):
        r = client.post("/score", json={**BASE_TXN, "timestamp": "yesterday-ish"}, headers=AUTH)
        assert r.status_code == 422

    def test_rapid_repeats_raise_velocity(self, client):
        card = {**BASE_TXN, "card_id": "card_velocity_test"}
        last = None
        for _ in range(6):
            last = client.post("/score", json=card, headers=AUTH).json()
        assert last["velocity"]["txn_count_1hr"] >= 5

    def test_internal_errors_do_not_leak(self, client, monkeypatch):
        import app.routers.score as score_module

        def boom(*a, **k):
            raise RuntimeError("secret internal detail")

        monkeypatch.setattr(score_module.fraud_scorer, "score", boom)
        r = client.post("/score", json=BASE_TXN, headers=AUTH)
        assert r.status_code == 500
        assert "secret internal detail" not in r.text


class TestRateLimit:
    def test_429_after_limit(self, client, fresh_rate_limiter):
        fresh_rate_limiter.max_requests = 3
        codes = [client.get("/health", headers=AUTH).status_code for _ in range(5)]
        assert codes[:3] == [200, 200, 200]
        assert codes[3] == 429

    def test_429_has_retry_after(self, client, fresh_rate_limiter):
        fresh_rate_limiter.max_requests = 1
        client.get("/health", headers=AUTH)
        r = client.get("/health", headers=AUTH)
        assert r.status_code == 429
        assert "retry-after" in r.headers


class TestAdjudicator:
    def test_gate_band(self):
        adj = Adjudicator(margin=0.05)
        assert adj.needs_review(0.78, 0.80)
        assert not adj.needs_review(0.50, 0.80)

    def test_disabled_falls_back_to_threshold(self):
        adj = Adjudicator(margin=0.05)
        adj.enabled = False
        out = adj.review(100.0, {}, {}, [], prob=0.82, threshold=0.80)
        assert out == {"decision": "BLOCK", "reasoning": None, "reviewed": False}

    def test_failure_fails_open(self):
        adj = Adjudicator(margin=0.05)
        adj.enabled = True

        def boom(**k):
            raise RuntimeError("sidecar down")

        adj._call_jev = boom
        out = adj.review(100.0, {}, {}, [], prob=0.78, threshold=0.80)
        assert out["decision"] == "ALLOW"
        assert out["reviewed"] is False


class TestMetrics:
    def test_metrics_exposed_without_auth(self, client):
        client.post("/score", json=BASE_TXN, headers=AUTH)
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "fraud_api_requests_total" in r.text
        assert "fraud_score_latency_seconds" in r.text

    def test_every_metric_the_dashboard_queries_is_exported(self, client):
        """Guards against dashboard/exporter drift (panels silently going empty)."""
        import re
        dash = (Path(__file__).parent.parent / "monitoring/grafana/dashboards/fraud_api.json").read_text()
        wanted = set(re.findall(r"((?:fraud|challenger)_[a-z0-9_]+?)(?:_bucket|_total)?(?=[\[{)\s\"])", dash))
        client.post("/score", json=BASE_TXN, headers=AUTH)
        text = client.get("/metrics").text
        exported = set(re.findall(r"^# TYPE (\S+)", text, re.M))
        base = {e.removesuffix("_total") for e in exported}
        missing = {w for w in wanted if w.removesuffix("_total") not in base}
        assert wanted, "no dashboard metrics parsed"
        assert not missing, missing


class TestChallengerAndModelInfo:
    def test_divergence_endpoint(self, client):
        client.post("/score", json=BASE_TXN, headers=AUTH)
        r = client.get("/challenger/divergence", headers=AUTH)
        assert r.status_code == 200
        assert "records" in r.json() and "count" in r.json()

    def test_divergence_requires_key(self, client):
        assert client.get("/challenger/divergence").status_code == 401

    def test_model_info(self, client):
        r = client.get("/model/info", headers=AUTH)
        assert r.status_code == 200
        assert "champion" in r.json()
