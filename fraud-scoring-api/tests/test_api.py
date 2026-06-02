"""
Tests for Fraud Scoring API.
Run: pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Mock models before importing app ──
mock_champion = MagicMock()
mock_champion.predict_proba.return_value = np.array([[0.85, 0.15]])

mock_challenger = MagicMock()
mock_challenger.predict_proba.return_value = np.array([[0.80, 0.20]])

mock_explainer = MagicMock()
mock_explainer.shap_values.return_value = np.array([[0.3, 0.1, 0.05, 0.02, 0.01,
                                                       0.08, 0.04, 0.02, 0.06, 0.03,
                                                       0.07, 0.09, 0.01, 0.02]])

with patch("app.main.load_models"):
    from app.main import app, champion_model, challenger_model

    import app.main as main_module
    main_module.champion_model = mock_champion
    main_module.challenger_model = mock_challenger
    main_module.explainer = mock_explainer
    main_module.model_meta = {
        "feature_cols": main_module.FEATURE_COLS,
        "card_avg": {"1234": 150.0},
        "card_std": {"1234": 50.0},
        "default_threshold": 0.42,
    }


@pytest.fixture
def client():
    return TestClient(app)


BASE_TXN = {
    "transaction_id": "TXN_TEST_001",
    "card_id": 1234,
    "transaction_amt": 450.0,
    "product_cd": "W",
    "transaction_dt": 1700000000,
    "balance_before": 1200.0,
    "balance_after": 750.0,
    "merchant_id": 555,
    "card_country": "US",
    "card_txn_count_24hr": 5,
    "merchant_txn_freq": 100,
}


class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "docs" in r.json()


class TestScoreEndpoint:
    def test_score_returns_200(self, client):
        r = client.post("/score", json=BASE_TXN)
        assert r.status_code == 200

    def test_score_response_schema(self, client):
        r = client.post("/score", json=BASE_TXN)
        body = r.json()
        assert "fraud_score" in body
        assert "risk_tier" in body
        assert "top_3_risk_factors" in body
        assert "model_used" in body
        assert "latency_ms" in body
        assert "rule_flags" in body

    def test_fraud_score_in_range(self, client):
        r = client.post("/score", json=BASE_TXN)
        score = r.json()["fraud_score"]
        assert 0.0 <= score <= 1.0

    def test_risk_tier_valid(self, client):
        r = client.post("/score", json=BASE_TXN)
        assert r.json()["risk_tier"] in ["low", "medium", "high", "critical"]

    def test_top_factors_count(self, client):
        r = client.post("/score", json=BASE_TXN)
        assert len(r.json()["top_3_risk_factors"]) <= 3

    def test_latency_ms_logged(self, client):
        r = client.post("/score", json=BASE_TXN)
        assert r.json()["latency_ms"] > 0


class TestRuleFlags:
    def test_foreign_txn_flagged(self, client):
        txn = {**BASE_TXN, "card_country": "NG"}
        r = client.post("/score", json=txn)
        assert "foreign_transaction" in r.json()["rule_flags"]

    def test_high_amt_flagged(self, client):
        txn = {**BASE_TXN, "transaction_amt": 9999.0}  # > 3x avg of 150
        r = client.post("/score", json=txn)
        assert "amt_exceeds_3x_card_avg" in r.json()["rule_flags"]

    def test_high_velocity_flagged(self, client):
        txn = {**BASE_TXN, "card_txn_count_24hr": 20}
        r = client.post("/score", json=txn)
        assert "high_velocity_24hr" in r.json()["rule_flags"]

    def test_normal_txn_no_flags(self, client):
        txn = {**BASE_TXN, "transaction_amt": 100.0, "card_country": "US"}
        mock_champion.predict_proba.return_value = np.array([[0.95, 0.05]])
        r = client.post("/score", json=txn)
        # Should have no rule flags for normal transaction
        flags = r.json()["rule_flags"]
        assert "foreign_transaction" not in flags


class TestThreshold:
    def test_threshold_returns_cutoff(self, client):
        r = client.post("/threshold?fpr=0.01")
        assert r.status_code == 200
        body = r.json()
        assert "calibrated_threshold" in body
        assert 0.0 < body["calibrated_threshold"] < 1.0
        assert body["requested_fpr"] == 0.01

    def test_threshold_custom_fpr(self, client):
        r = client.post("/threshold?fpr=0.05")
        assert r.status_code == 200
        assert r.json()["requested_fpr"] == 0.05


class TestMetrics:
    def test_prometheus_metrics_exposed(self, client):
        # Trigger a request first
        client.post("/score", json=BASE_TXN)
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "fraud_api_requests_total" in r.text
        assert "fraud_score_latency_seconds" in r.text


class TestChallengerDivergence:
    def test_divergence_endpoint(self, client):
        r = client.get("/challenger/divergence")
        assert r.status_code == 200
        assert "records" in r.json()
        assert "count" in r.json()


class TestRiskTierLogic:
    def test_tier_low(self):
        from app.main import get_tier
        assert get_tier(0.1) == "low"

    def test_tier_medium(self):
        from app.main import get_tier
        assert get_tier(0.45) == "medium"

    def test_tier_high(self):
        from app.main import get_tier
        assert get_tier(0.72) == "high"

    def test_tier_critical(self):
        from app.main import get_tier
        assert get_tier(0.95) == "critical"
