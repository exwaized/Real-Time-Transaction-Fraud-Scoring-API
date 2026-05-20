import logging, time
from fastapi import FastAPI
from app.routers.score import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S"
)

app = FastAPI(
    title="Real-Time Fraud Scoring API",
    description="Velocity-aware fraud detection with SMOTE+LightGBM and drift monitoring.",
    version="1.0.0",
)

app.include_router(router, tags=["Fraud Scoring"])

@app.get("/")
def root():
    return {"service": "fraud-scoring-api", "status": "running", "docs": "/docs"}
