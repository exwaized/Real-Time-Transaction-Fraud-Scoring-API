# CPU-only. Laya (~421M params, non-autoregressive) runs in well under a
# second per call on CPU, so this no longer needs a GPU host the way the
# SemIf/Qwen3.5-4B setup it replaced did.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only torch wheel first -- pip's default resolution for `laya`'s torch
# dependency would otherwise pull the much larger CUDA-enabled build, which
# is dead weight in a container with no GPU.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

EXPOSE 8600

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8600"]
