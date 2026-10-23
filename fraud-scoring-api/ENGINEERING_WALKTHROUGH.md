# Engineering Walkthrough — Fraud Scoring API

This document describes what is **actually implemented** in this repository today, as opposed to
what `README.md` describes (see [Docs vs. reality](#docs-vs-reality-important) — the README, tests,
Dockerfile and docker-compose describe an older/aspirational design that no longer matches the code).

---

## 1. What this service does

A synchronous FastAPI service that scores a single card transaction for fraud risk in real time:

```
POST /score  →  velocity features  →  LightGBM model  →  decision (ALLOW/BLOCK)
                                     →  drift monitor (async, background)
                                     →  behavioral profile
```

It is a **single-model** system (LightGBM only), with velocity-based feature engineering computed
in-process and a background drift-detection thread. There is no champion/challenger routing, no
SHAP, no XGBoost, and no `/metrics` endpoint in the running code, despite those appearing in
`README.md`, `requirements.txt`, `docker-compose.yml` and `tests/test_api.py`.

---

## 2. Request lifecycle (routing logic)

Entry point: `main.py` → `app/routers/score.py`.

```
Client
  │  POST /score  {card_id, merchant_id, amount, merchant_category, timestamp?}
  ▼
SecurityHeadersMiddleware        (main.py)  — adds HSTS/nosniff/X-Frame-Options/no-store on every response
  ▼
HTTPSRedirectMiddleware          (main.py)  — only mounted when ENV=production
  ▼
Depends(protected)               (score.py) — composite dependency, runs BEFORE the handler body:
  │   1. verify_api_key(X-API-Key header)   (app/services/auth.py)
  │   2. rate_limiter.check(api_key)        (app/services/rate_limit.py)
  ▼
score_transaction(txn)            (app/routers/score.py)
  │
  │  1. Parse timestamp → hour, day_of_week
  │  2. velocity_engine.get_velocity_features(card_id, amount)   — READ current rolling window
  │  3. fraud_scorer.score(amount, hour, dow, category, velocity) — model inference
  │  4. velocity_engine.record(card_id, amount, merchant_id, category) — WRITE, happens AFTER scoring
  │  5. drift_monitor.record(fraud_probability, velocity)         — feed background drift detector
  │  6. velocity_engine.get_profile(card_id)                      — behavioral aggregates for response
  ▼
FraudScoreResponse  {card_id, fraud_probability, decision, threshold_used, top_signals,
                      velocity, behavioral_profile, latency_ms}
```

Two more protected routes hang off the same `protected` dependency:
- `GET /health` — liveness + current model threshold + drift status + uptime.
- `GET /model/info` — dumps `app/models/meta.json` verbatim (threshold, feature list, ROC-AUC, feature importances).

`GET /` is the only unauthenticated route (root service banner pointing at `/docs`).

### Why velocity is read-then-written around scoring, not written-then-read

`score.py` deliberately computes velocity features **before** calling `velocity_engine.record()` for
the current transaction (see the `# 3. Record AFTER scoring` comment). If the current transaction
were recorded first, it would inflate its own `txn_count_1hr` / `amount_zscore` — a transaction would
always see itself as "one more in the last hour" and skew its own z-score. This ordering keeps
velocity features strictly historical relative to the transaction being scored.

---

## 3. Component responsibilities

| File | Responsibility |
|---|---|
| `main.py` | FastAPI app construction, security headers, conditional HTTPS redirect, router mount |
| `app/routers/score.py` | HTTP surface: `/score`, `/health`, `/model/info`; orchestrates services per request |
| `app/services/auth.py` | API key check (`X-API-Key` header vs. `FRAUD_API_KEY` env var) |
| `app/services/rate_limit.py` | Per-key sliding-window rate limiter (in-memory) |
| `app/services/velocity.py` | Per-card rolling transaction history, velocity features, behavioral profile |
| `app/services/scorer.py` | Loads `model.pkl`/`scaler.pkl`/`meta.json` once; assembles feature vector; runs inference; builds human-readable signal strings |
| `app/services/drift.py` | Background daemon thread; tracks score/feature distribution drift, PSI, alerting |
| `app/models/schemas.py` | Pydantic request/response contracts |
| `train.py` | Offline training pipeline that produces the three model artifacts under `app/models/` |
| `load_test.py` | Standalone latency benchmark client (stale — see below) |

All stateful services (`fraud_scorer`, `velocity_engine`, `drift_monitor`, `rate_limiter`,
`divergence_log`) are **module-level singletons**, instantiated once at import time and shared across
all requests handled by a given worker process. Historically this was all **per-process, in-memory
state**: running with `--workers > 1` (or multiple containers) gave each worker its own independent
velocity history, rate-limit counters, drift baseline, and divergence feed — a card's velocity seen
by worker A was invisible to worker B.

`divergence_log` (`app/services/divergence.py`) is now the one exception: when `REDIS_URL` is set it
stores records in a Redis list instead of an in-process `deque`, so `GET /challenger/divergence`
returns the same feed regardless of which worker or replica handled a given `/score` call — the
concrete demonstration that this service *can* run correctly behind multiple workers, not just one.
It falls back to the old in-process `deque` when `REDIS_URL` is unset (same "one env var is the
switch" pattern as `FRAUD_API_KEY` in `auth.py`), so local dev needs no Redis instance.
`docker-compose.yml` runs a `redis` service and points `fraud-api` at it via `REDIS_URL=redis://redis:6379/0`.

Velocity history, rate-limit counters, and drift baselines were **deliberately left per-process** —
see [§7](#7-notable-gaps--things-to-know-before-extending) for why, rather than migrating everything
to Redis in one pass.

---

## 4. Key engineering decisions

### 4.1 Fail-fast model loading
`Scorer.__init__` loads `model.pkl`, `scaler.pkl`, and `meta.json` with no fallback — if `train.py`
hasn't been run, the app **fails to start** rather than serving with a missing/default model. This
is intentional: a fraud scorer silently running without a real model is worse than a crash.

### 4.2 Threshold is data-derived, not hardcoded
`train.py` doesn't pick a fixed probability cutoff. It sweeps `precision_recall_curve` and selects
the threshold that maximizes F1, then persists it to `meta.json`. `scorer.py` reads that value at
startup (`self.threshold`) and applies it in `score()` (`decision = "BLOCK" if prob >= threshold else "ALLOW"`).
This decouples "what cutoff to use" (a training-time, data-dependent decision) from "how to apply it"
(a runtime constant), and `GET /model/info` / `GET /health` expose it for observability instead of
requiring a code change to inspect.

Note the currently trained threshold is **~0.984** with **ROC-AUC 1.0 / best-F1 1.0** — the synthetic
training data (`train.py`) draws fraud amounts from `exponential(800)` and legitimate amounts from
`exponential(120)`, which is almost perfectly separable by amount alone. This is expected/fine for a
demo pipeline but means the "1.0 AUC" is an artifact of synthetic data separability, not a claim about
real-world performance — worth flagging in any interview/demo context so it isn't read as an accuracy claim.

### 4.3 Auth fails open in dev, closed in prod
`verify_api_key()` returns a sentinel `"dev-mode-no-key-set"` when `FRAUD_API_KEY` isn't set in the
environment, rather than rejecting all requests — so the API is usable locally with zero config. The
moment `FRAUD_API_KEY` is set (e.g., in a deployed `.env`), the same code path enforces it strictly
(401 on missing/incorrect key). One env var is the entire auth on/off switch — simple, but be aware
it means **auth is silently disabled if the env var is merely absent**, not just if it's explicitly
turned off. There's no logged warning when the app is running key-less.

### 4.4 Rate limiting is composed with auth, not separate
`protected()` in `score.py` chains `verify_api_key` → `rate_limiter.check`, and every protected route
declares one dependency (`Depends(protected)`) instead of two. This keeps the "what does it take to
call this endpoint" story in one place, and guarantees rate limiting is always keyed by the
authenticated API key (never by IP), so limits track callers even behind a shared proxy/NAT.

### 4.5 Velocity engine: two locks, not one
`VelocityEngine` uses **separate locks** for the rolling transaction window (`_lock`) and the
behavioral profile aggregates (`_profile_lock`). Both are updated on every `record()` call, but they
protect logically independent state and are read by different callers (`get_velocity_features` vs.
`get_profile`). Splitting them avoids serializing profile reads behind window writes and vice versa.
Windows are `deque(maxlen=500)` per card — a fixed-size ring buffer used as a rough proxy for "last
~7 days" of history rather than a time-bounded structure, which trades exactness for O(1) memory
per card and avoids unbounded growth for high-frequency cards.

### 4.6 Drift monitoring is decoupled from the request path
`DriftMonitor` runs its own **daemon thread** (`_run` / `_check`) on a fixed interval
(`CHECK_INTERVAL_SEC = 60`), rather than checking for drift synchronously inside `/score`. The
request path only ever calls `record()` (an O(1) append to a bounded deque), so drift computation —
which is O(n) over up to 5000 samples plus a PSI calculation — never adds latency to a scoring
request. The first 200 recorded scores become the drift baseline automatically; alerts require both
50+ samples recorded **and** a baseline already set, so the monitor is silent during cold start.

Two drift signals are tracked independently and OR'd together for alerting:
- **Score-distribution drift**: absolute shift in mean `fraud_probability` vs. baseline (threshold 0.15).
- **Feature PSI** on `amount_zscore` (Population Stability Index, threshold 0.20) — a standard
  credit-risk/fraud industry metric for detecting feature distribution shift between two populations.

Alerting is best-effort: `_fire_slack` swallows all exceptions and only logs a warning, so a broken
webhook never affects scoring or crashes the monitor thread. **However**, `drift_monitor = DriftMonitor()`
in `drift.py` is instantiated with no arguments, so `slack_webhook` is always `None` in the current
wiring — the `SLACK_WEBHOOK_URL` env var documented in `.env.example` is not actually read anywhere.
Drift alerts currently only reach `logger.critical(...)`, not Slack, until that's wired up.

### 4.7 Security headers as middleware, not per-route
`SecurityHeadersMiddleware` in `main.py` is a `BaseHTTPMiddleware` applied globally, guaranteeing
`X-Content-Type-Options`, `X-Frame-Options`, HSTS, `Cache-Control: no-store`, and
`Referrer-Policy` on every response — including error responses — without relying on every route
author to remember to add them. `HTTPSRedirectMiddleware` is conditionally mounted only when
`ENV=production`, so local HTTP development isn't forced onto HTTPS.

### 4.8 Signal generation is rule-based post-hoc explanation, not model explainability
`Scorer._build_signals()` does **not** use SHAP or any model-internal feature attribution. It's a
fixed set of hand-written heuristics over the velocity features and z-score (deviation ≥2σ, ≥5 txns/hr,
≥4 unique merchants/hr, repeat txn <30s apart), returned in a fixed priority order and truncated to
the top 3, with a fallback string (`fraud_probability=...`) if nothing else triggers. This is cheap,
deterministic, and human-readable, but it explains "what velocity looked suspicious," not "what the
model actually weighted" — a real explainability layer (SHAP, as the README implies) would need to
attribute the LightGBM prediction itself.

---

## 5. Data flow: training → serving contract

`train.py` and `app/services/scorer.py` must agree on feature order and encoding — this is the
single most fragile coupling in the codebase, held together by convention rather than a shared
module:

```
train.py: FEATURES = ["amount","hour","day_of_week"] + ["cat_electronics", ..., "cat_pharmacy"]
scorer.py: FEATURES = [same list, duplicated]           ← comment: "must match train.py exactly"
```

Both independently one-hot encode `merchant_category` against the same 8-category list
(`CATEGORIES` in `scorer.py` / `cats` in `train.py`). If a new category is added in one file and not
the other, or the list order changes, the scaler/model will silently score against misaligned
columns — no runtime check enforces this contract; it relies on the inline comment and file-parity
between the two files.

Artifacts produced by `train.py` and consumed by `scorer.py`:

| File | Producer | Consumer | Contents |
|---|---|---|---|
| `app/models/model.pkl` | `train.py` (LightGBM, SMOTE-resampled) | `Scorer.__init__` | trained classifier |
| `app/models/scaler.pkl` | `train.py` (`StandardScaler`) | `Scorer.__init__` | feature scaling, fit on resampled training data |
| `app/models/meta.json` | `train.py` | `Scorer.__init__`, `/model/info`, `/health` | threshold, feature list, ROC-AUC, best-F1, per-feature importances |

SMOTE is applied **after** the train/test split and only to the training fold — a correct choice
that avoids leaking synthetic minority-class samples into the held-out evaluation set.

---

## 6. Docs vs. reality (important)

`README.md`, `tests/test_api.py`, `Dockerfile`, `docker-compose.yml`, `requirements.txt`, and
`load_test.py` describe a **different, more elaborate system** than what's implemented in `main.py` /
`app/`. This looks like leftover scaffolding from an earlier design iteration (the README's "Amex CFR
Alignment" section suggests this project was built/aimed at a specific job application). Concretely:

| Described in docs/tests/config | Actually implemented |
|---|---|
| Champion (LightGBM) + Challenger (XGBoost), 90/10 traffic split | Single LightGBM model only |
| SHAP-based `top_3_risk_factors` | Rule-based `top_signals` heuristics, no SHAP |
| `POST /threshold?fpr=...` endpoint | Does not exist |
| `GET /challenger/divergence` endpoint | Does not exist |
| `GET /metrics` (Prometheus) endpoint | Does not exist — `prometheus-client` is in `requirements.txt` but never imported anywhere in `app/` |
| MLflow model registry | Not used anywhere in code; only appears as a docker-compose service |
| Request schema: `transaction_id`, `card_id` (int), `transaction_amt`, `product_cd`, `balance_before/after`, `card_country`, `card_txn_count_24hr`, `merchant_txn_freq` | Actual schema (`app/models/schemas.py`): `card_id` (str), `merchant_id`, `amount`, `merchant_category`, `timestamp` |
| Response: `fraud_score`, `risk_tier` (low/med/high/critical), `rule_flags`, `model_used` | Actual response: `fraud_probability`, `decision` (ALLOW/BLOCK), `top_signals`, `velocity`, `behavioral_profile` |
| `tests/test_api.py` imports `app.main.load_models`, `champion_model`, `challenger_model`, `get_tier` | None of these exist; `app/main.py` doesn't even exist (the app is `main.py` at repo root) — **the entire test suite fails to collect against the current code** |
| `Dockerfile` does `COPY app/ ./app/` + `COPY models/ ./models/`, runs `uvicorn app.main:app` | No top-level `models/` dir (artifacts live at `app/models/`) and no `app/main.py` — **image would build but the container would crash on start** |
| `docker-compose.yml` sets `CHALLENGER_RATIO=0.10`, mounts `./models:/app/models:ro` | `CHALLENGER_RATIO` is read nowhere in code; volume mount path doesn't match where the app actually looks for models |
| `load_test.py` sends the old `transaction_amt`/`card_country` schema, no `X-API-Key` header | Would get 422 (schema mismatch) and/or 401 (missing auth) against the current API, not real latency numbers |

**Practical implication:** if you run `docker-compose up --build` or `pytest tests/ -v` right now,
expect them to fail or misbehave — they're testing/packaging a prior version of this service. Anyone
picking up this repo should treat `main.py` + `app/` as the source of truth and either regenerate
`tests/`, `Dockerfile`, `docker-compose.yml`, and `load_test.py` to match the current contract, or
explicitly scope work to bringing the docs/tests back in sync before relying on them.

---

## 7. Notable gaps / things to know before extending

- **No persistence for velocity/rate-limit/drift** — velocity history, behavioral profiles, drift
  baselines, and rate-limit windows all live in process memory and are lost on restart, and are not
  shared across workers/replicas. The divergence log (§3) moved to Redis as a proof-of-concept for
  what a real multi-worker rollout requires; the other three were left in-process on purpose:
  - `rate_limiter` and `velocity_engine` sit in the **hot path of every `/score` call** — moving them
    to Redis trades an in-memory deque append for a network round-trip on every request, and the rate
    limiter in particular would need a Lua script or `MULTI`/`WATCH` to keep the check-then-increment
    atomic across workers (a plain read-then-write race would let concurrent requests both slip past
    the limit).
  - `drift_monitor`'s baseline is a statistical property of *one* worker's sample stream; averaging
    it across workers needs either a shared streaming-stats structure or a moved-out drift-computation
    step, not a drop-in Redis swap.
  - `divergence_log` doesn't sit in a decision path — it's a best-effort side log (same as the Slack
    drift-alert webhook), so a network write that fails or adds a few ms of latency doesn't affect
    the score or block the response. That made it the lowest-risk place to demonstrate shared state.
- **No `/metrics` endpoint** despite Prometheus being configured to scrape one (`monitoring/prometheus/prometheus.yml`
  targets `fraud-api:8000/metrics`) — the scrape will 404 against the current app.
- **Slack drift alerting is dead code** — `slack_webhook` is never passed to `DriftMonitor()`, so
  `SLACK_WEBHOOK_URL` from `.env.example` has no effect currently.
- **No schema-parity check** between `train.py`'s feature list and `scorer.py`'s — a manual/convention
  contract, not enforced in code or tests.
- **Auth silently no-ops without `FRAUD_API_KEY`** — safe default for local dev, but there's no startup
  log line calling this out, so it could go unnoticed in a real deployment.
