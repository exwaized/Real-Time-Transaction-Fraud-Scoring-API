# Requires an NVIDIA GPU + the nvidia-container-toolkit on the host
# (docker run --gpus all ...). Not something the main fraud-scoring-api
# image needs — that's the whole point of splitting this out.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

EXPOSE 8600

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8600"]
