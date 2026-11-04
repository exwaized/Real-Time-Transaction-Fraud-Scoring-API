# ── Stage 1: train the champion/challenger models ─────────────────────────────
# Model artifacts are gitignored, so a clean checkout (CI, Render, ...) has none.
# train.py is seeded (seed 42), so every build produces the same models.
FROM python:3.11-slim AS train

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

COPY train.py .
RUN python train.py


# ── Stage 2: slim serving image ───────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY app/ ./app/
COPY main.py .
# scorer.py loads models from app/models relative to itself (/app/app/models)
COPY --from=train /build/app/models/ ./app/models/

ENV PYTHONUNBUFFERED=1
ENV PROMETHEUS_MULTIPROC_DIR=/tmp/prom_multiproc
RUN mkdir -p /tmp/prom_multiproc && useradd -r -u 10001 appuser \
    && chown -R appuser /tmp/prom_multiproc /app
USER appuser

EXPOSE 8000

# PORT is injected by hosts like Render; WEB_CONCURRENCY should be 1 where there
# is no REDIS_URL, because velocity windows and rate limits are per-process then.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WEB_CONCURRENCY:-2}"]
