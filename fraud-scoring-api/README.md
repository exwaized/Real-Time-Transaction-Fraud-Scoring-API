# Real-Time Transaction Fraud Scoring API
**Production-grade fraud detection system — architected for FRM roles **

---

## Quick Start

```bash
# 1. Configure secrets (compose refuses to start without these two)
cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(32))"   # use for FRAUD_API_KEY
python -c "import secrets; print(secrets.token_urlsafe(16))"   # use for GRAFANA_ADMIN_PASSWORD
# edit .env: set FRAUD_API_KEY and GRAFANA_ADMIN_PASSWORD

# 2. Train models (writes app/models/*.pkl + meta.json; these are gitignored,
#    so every fresh clone must run this once)
pip install -r requirements.txt
python train.py

# 3. Launch the stack (API + Redis + Prometheus + Grafana + MLflow + sidecar)
docker compose up -d --build

# 4. Score a transaction
curl -X POST http://localhost:8000/score   -H "Content-Type: application/json"   -H "X-API-Key: $FRAUD_API_KEY"   -d '{
    "card_id": "card_0042",
    "merchant_id": "merch_017",
    "amount": 1800.00,
    "merchant_category": "electronics",
    "timestamp": "2025-05-19T03:22:11"
  }'

# 5. Run tests (needs the trained models from step 2)
pytest tests/ -v

# 6. Benchmark latency
python load_test.py
```

Auth fails closed: if no API key is configured the API returns 503 on every
protected route. Prefer one key per caller (`FRAUD_API_KEYS=payments=<key>,risk-ui=<key>`)
with per-client limits (`FRAUD_RATE_LIMITS=payments=600`, default 60 req/min); a single
shared `FRAUD_API_KEY` still works as client `default`. Set `CARD_HASH_SALT` to a long
random value so card IDs are hashed in logs and Redis keys. For local non-Docker development only, set
`ALLOW_INSECURE_NO_AUTH=true` to bypass it.

**Endpoints & dashboards:**
| Service | URL | Credentials |
|---|---|---|
| API Docs | http://localhost:8000/docs | — |
| Liveness probe | http://localhost:8000/healthz | — (no key) |
| Metrics | http://localhost:8000/metrics | — (keep internal; don't expose publicly) |
| Grafana | http://localhost:3000 | `admin` / your `GRAFANA_ADMIN_PASSWORD` |
| Prometheus | http://localhost:9090 | — |
| MLflow | http://localhost:5000 | — |

Grafana, Prometheus, Redis and MLflow bind to localhost only.

**Alerts** (`monitoring/prometheus/alerts.yml`, visible at http://localhost:9090/alerts; wire an
Alertmanager to get paged): API down, Redis degraded (`fraud_api_shared_state_degraded == 1`),
>1% 500s, p99 latency >500ms, adjudicator fallbacks.

---

## Architecture

```
POST /score
   │
   ├─ API key check (per-client) ─► rate limit (per client, Redis or in-process)
   │
   ├─ velocity features per card (1h/24h counts, 7-day mean and z-score)
   │
   ├─ Champion (LightGBM) and Challenger (XGBoost) both score
   │     └─ ~10% of traffic is routed to the challenger (CHALLENGER_RATIO)
   │
   ├─ threshold decision: ALLOW / BLOCK
   │     └─ optional adjudicator reviews scores near the threshold
   │        (fails open to the raw decision)
   │
   └─ response + Prometheus metrics + divergence log

Background: drift monitor (score shift and PSI) logs CRITICAL and can post to Slack.
Shared state (velocity, profiles, rate limits) uses Redis when REDIS_URL is set;
otherwise each worker keeps its own, so run a single worker.
```

---

## Model Card

> **Demo data, not a benchmark.** `train.py` generates 10,000 synthetic card
> transactions (3% fraud, seed 42). Fraud and legitimate rows are drawn from
> clearly different distributions (fraud: larger amounts, night hours,
> electronics/online/ATM), so both models separate them almost perfectly on the
> 20% held-out split. The point of this repo is the serving, monitoring and
> safety layers around a model, not model quality. Real transaction data would
> score far lower.

| | Champion | Challenger |
|---|---|---|
| Algorithm | LightGBM, SMOTE on the training split | XGBoost, `scale_pos_weight` |
| ROC-AUC (held-out 20%) | 1.000 | 1.000 |
| Decision threshold (F1-optimal) | 0.984 | 0.999 |

Metrics come from `train.py` and are written to `app/models/meta.json`
(`GET /model/info`). Latency depends on hardware: run `load_test.py` to measure it.
In a local Docker run the first request after startup took about 46 ms.
`train.py` picks each threshold on the same held-out split it reports on, so
treat those numbers as optimistic; the real-data benchmark below picks the
threshold on a separate validation slice.

### Real-data benchmark (ULB credit card fraud)

The same recipe (LightGBM + SMOTE champion, XGBoost challenger) run on a real
public dataset: the ULB / Worldline European card transactions from Kaggle
([mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)),
284,807 transactions over two days, 0.17% fraud. Features are anonymised PCA
components (`V1`..`V28`) plus `Amount`. The data is not stored in this repo;
see the Kaggle page for its licence and citation.

Split by time: first 60% train, next 20% validation (threshold chosen here),
last 20% test (scored once). The test slice holds 56,962 transactions, 75 of
them fraud.

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Recall at 1% FPR |
|---|---|---|---|---|---|---|
| Champion: LightGBM + SMOTE | 0.960 | **0.786** | 0.918 | 0.747 | **0.824** | 0.840 |
| Challenger: XGBoost | 0.981 | 0.779 | 0.883 | 0.707 | 0.785 | 0.853 |
| Baseline: logistic regression | 0.982 | 0.742 | 0.943 | 0.667 | 0.781 | 0.867 |

How to read it:
- With only 75 fraud cases in the test slice, one more or fewer caught changes
  recall by about 1.3 points. The three models are closer than the table makes
  them look, and the plain logistic regression baseline is competitive.
- PR-AUC is the more useful number at this imbalance; ROC-AUC flatters all three.
- These features are anonymised, so this model cannot be served through `/score`,
  which takes card, merchant, amount and category. The served model is still the
  synthetic demo above; this benchmark shows the training recipe holds up on real data.

Reproduce it (full output in `benchmarks/results_ulb_creditcard.json`):

```bash
python benchmarks/real_data_benchmark.py /path/to/creditcard.csv --out results.json
```

---

## Features

Model inputs (`app/services/scorer.py`): `amount`, `hour`, `day_of_week`, and a
one-hot `merchant_category` (electronics, online, atm, travel, grocery, fuel,
restaurant, pharmacy).

Velocity signals are computed per card at request time
(`app/services/velocity.py`) and returned in `top_signals` / `velocity`. They
explain a decision but are not inputs to the model: 1h and 24h transaction
counts, unique merchants in 1h, 7-day mean and z-score of the amount, and time
since the last transaction.

---

## API Reference

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/score` | `X-API-Key` | Score a transaction |
| GET | `/healthz` | none | Liveness probe |
| GET | `/health` | `X-API-Key` | Model threshold, drift status, uptime |
| GET | `/model/info` | `X-API-Key` | Training metadata and feature importances |
| GET | `/challenger/divergence` | `X-API-Key` | Last 100 champion vs challenger deltas |
| GET | `/metrics` | none | Prometheus metrics (keep internal) |

### `POST /score`
```json
Request:
{
  "card_id": "card_0042",
  "merchant_id": "merch_017",
  "amount": 12400.00,
  "merchant_category": "electronics",
  "timestamp": "2025-05-19T03:22:11"
}

Response (abridged):
{
  "card_id": "card_0042",
  "fraud_probability": 1.0,
  "champion_probability": 1.0,
  "challenger_probability": 0.9999,
  "model_used": "champion_lgbm",
  "decision": "BLOCK",
  "threshold_used": 0.984,
  "top_signals": ["fraud_probability=1.0000 (model score above threshold)"],
  "velocity": {"txn_count_1hr": 0, "amount_zscore": 0.0, "...": "..."},
  "latency_ms": 46.25,
  "reviewed": false
}
```

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
| Requests | `fraud_api_requests_total` (by outcome) |
| Errors | `fraud_api_errors_total` |
| Latency | `fraud_score_latency_seconds` |
| Score distribution | `fraud_score_distribution` histogram |
| Risk signals | `fraud_api_signals_total{signal}` |
| Model divergence | `challenger_score_divergence` histogram |
| Adjudicator fallbacks | `fraud_api_adjudicator_fallbacks_total` |
| Redis degraded | `fraud_api_shared_state_degraded` (1 = per-process fallback) |
| Drift | Score-mean shift > 0.15 or PSI > 0.20 logs CRITICAL, and posts to Slack if `SLACK_WEBHOOK_URL` is set |


---

## Project Structure

```
fraud-scoring-api/
├── train.py                    # trains champion + challenger on synthetic data
├── main.py                     # FastAPI app
├── load_test.py                # latency benchmark
├── requirements.txt            # full dev/CI set
├── requirements-serve.txt      # runtime set used by the Docker image
├── requirements-train.txt      # training stage of the Docker build
├── Dockerfile                  # multi-stage: train, then slim serving image
├── docker-compose.yml          # API + Redis + Prometheus + Grafana + sidecar
├── app/
│   ├── routers/score.py        # endpoints
│   ├── services/               # scorer, velocity, drift, auth, rate limit, redis, metrics
│   └── models/                 # schemas, plus .pkl artifacts generated by train.py
├── tests/test_api.py           # pytest suite
├── monitoring/                 # Prometheus + Grafana config
└── semif_sidecar/              # optional borderline-score adjudicator
```
