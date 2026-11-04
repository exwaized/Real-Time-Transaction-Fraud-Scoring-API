# Real-Time Transaction Fraud Scoring API

A production-grade fraud detection service with velocity feature engineering,
SMOTE-balanced classification, tuned decision thresholds, and live drift monitoring.

Real-time fraud scoring API that detects card-testing bursts, amount anomalies, 
and behavioral deviations using per-card velocity features and a SMOTE-balanced 
LightGBM classifier. Exposes a FastAPI POST /score endpoint (<50ms) returning 
fraud probability, BLOCK/ALLOW decision, and top-3 signals. Includes a background 
drift monitor that tracks score distribution and feature PSI, firing Slack alerts 
on model degradation.

## Project Structure

```
fraud-api/
├── app/
│   ├── main.py                  # FastAPI app
│   ├── train.py                 # SMOTE + LightGBM training
│   ├── models/
│   │   ├── schemas.py           # Pydantic request/response models
│   │   ├── model.pkl            # Trained LightGBM (generated)
│   │   ├── scaler.pkl           # StandardScaler (generated)
│   │   └── meta.json            # Threshold, AUC, feature importances
│   ├── routers/
│   │   └── score.py             # POST /score, GET /health, GET /model/info
│   └── services/
│       ├── velocity.py          # Thread-safe rolling window store
│       ├── scorer.py            # Feature assembly + model inference
│       └── drift.py             # Background PSI + score drift monitor
├── data/
│   ├── generate_data.py         # Synthetic 10K transaction dataset
│   └── transactions.csv         # Generated dataset (3% fraud rate)
└── requirements.txt
```

---

## Setup & Run

```bash
pip install -r requirements.txt
python data/generate_data.py
python app/train.py
uvicorn app.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

---

## Sample Request

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "card_id": "card_0042",
    "merchant_id": "merch_017",
    "amount": 12400.00,
    "merchant_category": "electronics",
    "timestamp": "2025-05-19T03:22:11"
  }'
```

**Response:**
```json
{
  "card_id": "card_0042",
  "fraud_probability": 0.9991,
  "decision": "BLOCK",
  "threshold_used": 0.433,
  "top_signals": [
    "amount_zscore=11900000000000.00 (≥2σ deviation from card history)",
    "txn_count_1hr=8 (high velocity burst)",
    "unique_merchants_1hr=8 (card-testing pattern)"
  ],
  "velocity": {
    "txn_count_1hr": 8,
    "txn_count_24hr": 8,
    "amount_mean_7d": 500.0,
    "amount_zscore": 11900000000000.0,
    "unique_merchants_1hr": 8,
    "time_since_last_txn_sec": 0.0
  },
  "behavioral_profile": {
    "avg_amount": 500.0,
    "top_hour": 16,
    "top_category": "electronics",
    "total_txns_seen": 8
  },
  "latency_ms": 9.34
}
```

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /score | Score a transaction |
| GET | /health | Model status + drift monitor state |
| GET | /model/info | Feature importances + training metadata |

---

## Model Performance

### Real data (measured)

Trained and evaluated on the real **ULB credit-card fraud dataset** (OpenML 1597):
284,807 transactions, 9,144 exact duplicates removed -> 275,663 unique, 473 frauds (0.17%).
Stratified 64/16/20 train/val/test split; the decision threshold is tuned on the validation
slice and frozen before scoring the test set. No SMOTE and no class re-weighting
(re-weighting with `scale_pos_weight` cut PR-AUC from ~0.86 to ~0.24 in testing).

| Metric (held-out test set, 55,133 txns, 95 frauds) | Value |
|---|---|
| ROC-AUC | 0.978 |
| PR-AUC (average precision) | 0.822 |
| Precision at tuned threshold (0.2017) | 0.901 |
| Recall at tuned threshold | 0.768 (73 of 95 frauds caught) |
| F1 | 0.830 |
| False alarms | 8 legitimate transactions flagged |
| Model inference latency (single row, CPU) | p50 0.84 ms, p95 1.2 ms |

Reproduce: `cd fraud-scoring-api && python train_real.py` then call `POST /score/real`
with `amount` and the 28 PCA components `v` (V1..V28).

Caveats: the OpenML copy has no `Time` column, so the split is random rather than
chronological and scores may be slightly optimistic versus a time-ordered split. The
features are anonymised PCA components, so the card-velocity features of the demo
`/score` endpoint do not apply to this model. Latency is model-only; end-to-end API
latency is higher.

### Demo model on synthetic data (NOT real performance)

`POST /score` uses a model trained by `train.py` on a generated 10,000-transaction
dataset (3% fraud). Its metrics (ROC-AUC 0.989, F1 0.862, ~9 ms) describe how well it
separates the generator's own fraud patterns, not real-world fraud, and should not be
quoted as real performance.

## Architecture Notes

- **VelocityEngine**: reuses RLock + deque pattern from Distributed Anomaly Detection Engine
- **DriftMonitor**: daemon thread computing PSI on `amount_zscore` + mean score shift every 60s
- **Scorer**: loads pickle artifacts at startup, single-instance across all workers
- **Behavioral profiles**: incremental update on every `record()` call, O(1) amortized
