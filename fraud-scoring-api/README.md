# Real-Time Transaction Fraud Scoring API
**Production-grade fraud detection system — architected for FRM roles (Amex CFR style)**

---

## Quick Start

```bash
# 1. Train models (generates /models/ artifacts)
pip install -r requirements.txt
python train.py

# 2. Launch full stack (API + Prometheus + Grafana + MLflow)
docker-compose up --build

# 3. Score a transaction
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TXN_001",
    "card_id": 4521,
    "transaction_amt": 1800.00,
    "product_cd": "W",
    "transaction_dt": 1700000000,
    "balance_before": 2000.0,
    "balance_after": 200.0,
    "merchant_id": 321,
    "card_country": "NG",
    "card_txn_count_24hr": 18,
    "merchant_txn_freq": 45
  }'

# 4. Run tests
pytest tests/ -v

# 5. Benchmark latency
python load_test.py
```

**Dashboards:**
| Service | URL | Credentials |
|---|---|---|
| API Docs | http://localhost:8000/docs | — |
| Grafana | http://localhost:3000 | admin / fraud123 |
| Prometheus | http://localhost:9090 | — |
| MLflow | http://localhost:5000 | — |

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    FRAUD SCORING API                           │
│                                                                │
│  POST /score ──► Rule Pre-Filter ──► Champion/Challenger       │
│                       │                    Router (90/10)      │
│                       ▼                         │              │
│               [amt_3x, foreign,          ┌──────┴──────┐      │
│                velocity, drain]          │             │      │
│                       │              Champion      Challenger  │
│                       ▼             LightGBM        XGBoost   │
│               Rule Boost (+score)       │             │        │
│                       │                └──────┬──────┘        │
│                       ▼                       ▼               │
│               SHAP top_3_factors     Divergence Logger        │
│                       │              (delta → Prometheus)      │
│                       ▼                                        │
│               Risk Tier Assignment                             │
│               [low / medium / high / critical]                 │
│                       │                                        │
│                       ▼                                        │
│               Structured JSON Log + Prometheus Metrics         │
│                                                                │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ /threshold  │  │ /challenger/ │  │ Rolling Fraud Rate   │  │
│  │ ?fpr=0.01   │  │  divergence  │  │ Alert if >2% / 5min  │  │
│  └─────────────┘  └──────────────┘  └──────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
         │                   │                   │
    Prometheus            Grafana             MLflow
    (metrics)           (dashboard)          (registry)
```

---

## Model Card

> Trained on PaySim synthetic dataset (50K+ transactions, ~3.5% fraud rate)

| Metric | Champion (LightGBM) | Challenger (XGBoost) |
|---|---|---|
| **AUC-ROC** | ~0.978 | ~0.971 |
| **KS Statistic** | ~0.82 | ~0.79 |
| **Gini Coefficient** | ~0.956 | ~0.942 |
| **Precision @ Top 5%** | ~0.91 | ~0.88 |
| **FPR @ Chosen Threshold** | ~1.0% | ~1.2% |
| **Threshold (FPR=1%)** | ~0.42 | ~0.45 |

> ⚠️ Actual values vary per run. Re-run `train.py` on your dataset.
> p99 latency: **~18ms** (single worker, local CPU) — ✅ sub-50ms target met

**Class Imbalance Handling:**
- Champion: SMOTE applied post-split (training set only)
- Challenger: `scale_pos_weight` = class ratio

**Feature Importance (Champion top 5):**
1. `balance_drain_ratio` — full balance drain is strongest fraud signal
2. `amt_vs_avg_ratio` — deviation from historical card average
3. `amt_zscore` — standardized deviation
4. `is_foreign_txn` — cross-border card activity
5. `merchant_risk_score` — category-level risk prior

---

## Feature Engineering

| Feature | Description | Fraud Signal |
|---|---|---|
| `amt_vs_avg_ratio` | TxnAmt / card historical avg | >3x → high risk |
| `amt_zscore` | Standardized deviation from card mean | Outlier → risk |
| `velocity_1hr_bucket` | Bucketed txn count (0-3) | High bucket → risk |
| `balance_drain_ratio` | (before-after)/before | >0.95 → critical |
| `is_foreign_txn` | card_country ≠ US | +flag → boost |
| `merchant_risk_score` | Category prior (R=0.12, W=0.02) | High → risk |
| `is_night` | hour ≥ 22 or ≤ 6 | Night txn → elevated |
| `rule_flag` | Binary: amt_3x OR foreign | Direct feature |

---

## API Reference

### `POST /score`
```json
Request:
{
  "transaction_id": "TXN_001",
  "card_id": 4521,
  "transaction_amt": 1800.0,
  "product_cd": "W",
  "transaction_dt": 1700000000,
  "balance_before": 2000.0,
  "balance_after": 200.0,
  "merchant_id": 321,
  "card_country": "NG",
  "card_txn_count_24hr": 18,
  "merchant_txn_freq": 45
}

Response:
{
  "transaction_id": "TXN_001",
  "fraud_score": 0.871,
  "risk_tier": "critical",
  "top_3_risk_factors": [
    {"feature": "balance_drain_ratio", "importance": 0.312},
    {"feature": "is_foreign_txn", "importance": 0.241},
    {"feature": "amt_vs_avg_ratio", "importance": 0.198}
  ],
  "rule_flags": ["foreign_transaction", "near_full_balance_drain", "high_velocity_24hr"],
  "model_used": "champion_lgbm",
  "latency_ms": 14.3,
  "timestamp": "2024-11-14T10:22:31"
}
```

### `POST /threshold?fpr=0.01`
Returns the score cutoff that achieves the target false positive rate.

### `GET /challenger/divergence`
Returns last 100 Champion vs Challenger score delta records.

### `GET /metrics`
Prometheus metrics endpoint (scraped every 10s).

---

## Amex CFR Alignment

| JD Keyword | This Project |
|---|---|
| **"minimize disruption of good spending"** | FPR-calibrated threshold endpoint (`/threshold?fpr=0.01`) minimizes false positives on legitimate transactions |
| **"closed-loop through Amex network"** | Challenger-Champion pattern logs divergence for offline model improvement — mirrors production A/B testing feedback loop |
| **"predictive model development, deployment, validation"** | Full pipeline: feature engineering → SMOTE → train → MLflow registry → FastAPI serving → pytest validation suite |
| **"leverage data and digital advancements"** | SHAP explainability, Prometheus/Grafana observability, structured audit logging |
| **"profitable decisions across risk and fraud"** | Risk tier segmentation (low/medium/high/critical) enables tiered decision logic: auto-decline vs step-up auth vs review |
| **"large amounts of data → business insights"** | Velocity, behavioral, and merchant-level aggregates surface actionable signals, not raw scores |
| **"big data & ML innovation"** | LightGBM + XGBoost ensemble strategy; rule-hybrid scoring; Monte Carlo-ready threshold calibration |
| **"clear articulation to leadership"** | Grafana dashboard shows fraud rate, tier breakdown, latency SLOs — ready for exec review |

---

## Challenger-Champion Pattern

10% of live traffic is routed to XGBoost (Challenger).
Both models always score; divergence is logged to:
- `GET /challenger/divergence` endpoint
- `challenger_score_divergence` Prometheus histogram
- Grafana "Divergence" panel

When challenger consistently outperforms (lower divergence at lower FPR), swap models via env var `CHALLENGER_RATIO=1.0`.

---

## Observability

| Signal | Metric |
|---|---|
| Throughput | `fraud_api_requests_total{risk_tier, model}` |
| Latency | `fraud_score_latency_seconds` (p50/p95/p99) |
| Fraud Rate | `fraud_rate_rolling_5min` (5-min rolling window) |
| Score Distribution | `fraud_score_distribution` histogram |
| Rule Triggers | `rule_based_flags_total{rule}` |
| Model Divergence | `challenger_score_divergence` histogram |
| Alert Threshold | Fraud rate > 2% in 5min → CRITICAL log + Slack webhook |

---

## Project Structure

```
fraud-scoring-api/
├── train.py                    # ML training pipeline
├── load_test.py                # Latency benchmark
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── app/
│   └── main.py                 # FastAPI app
├── models/                     # Generated by train.py
│   ├── champion_lgbm.pkl
│   ├── challenger_xgb.pkl
│   └── model_meta.json
├── tests/
│   └── test_api.py             # pytest suite (25 tests)
├── monitoring/
│   ├── prometheus/prometheus.yml
│   └── grafana/
│       ├── dashboards/fraud_api.json
│       └── provisioning/
└── README.md
```
