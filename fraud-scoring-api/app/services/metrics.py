"""
Prometheus metrics. With multiple uvicorn workers, set PROMETHEUS_MULTIPROC_DIR
(the Dockerfile does) so /metrics aggregates across worker processes.
"""
import os
from prometheus_client import (
    CollectorRegistry, Counter, Gauge, Histogram, generate_latest,
    CONTENT_TYPE_LATEST, REGISTRY, multiprocess,
)

REQUESTS = Counter(
    "fraud_api_requests_total", "Scoring requests by outcome",
    ["decision", "model_used"],
)
ERRORS = Counter("fraud_api_errors_total", "Unhandled scoring errors")
ADJUDICATOR_FALLBACKS = Counter(
    "fraud_api_adjudicator_fallbacks_total",
    "Borderline reviews that fell back to the threshold decision",
)
SCORE_DISTRIBUTION = Histogram(
    "fraud_score_distribution", "Distribution of routed fraud probabilities",
    buckets=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
CHALLENGER_DIVERGENCE = Histogram(
    "challenger_score_divergence", "|champion - challenger| probability delta",
    buckets=(0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0),
)
SIGNALS = Counter("fraud_api_signals_total", "Risk signals emitted", ["signal"])
# 1 if Redis is configured but unavailable and shared state has degraded to
# per-process, else 0. "max" across live workers, so one degraded worker shows
# as 1. Initial value 0 = healthy, so a worker that hasn't touched Redis yet
# (or REDIS_URL unset) never looks degraded.
SHARED_STATE_DEGRADED = Gauge(
    "fraud_api_shared_state_degraded",
    "1 if Redis-backed shared state has degraded to per-process state",
    multiprocess_mode="max",
)
SCORE_LATENCY = Histogram(
    "fraud_score_latency_seconds", "End-to-end /score handler latency",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)


def render() -> tuple[bytes, str]:
    if os.getenv("PROMETHEUS_MULTIPROC_DIR"):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
    else:
        registry = REGISTRY
    return generate_latest(registry), CONTENT_TYPE_LATEST
