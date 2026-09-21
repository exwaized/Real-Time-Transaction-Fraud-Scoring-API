from pydantic import BaseModel, Field
from typing import Optional, List

class TransactionRequest(BaseModel):
    card_id: str = Field(..., example="card_0042")
    merchant_id: str = Field(..., example="merch_017")
    amount: float = Field(..., gt=0, example=12400.00)
    merchant_category: str = Field(..., example="electronics")
    timestamp: Optional[str] = Field(default=None, example="2025-05-19T03:22:11")

class FraudScoreResponse(BaseModel):
    card_id: str
    fraud_probability: float
    decision: str
    threshold_used: float
    top_signals: List[str]
    velocity: dict
    behavioral_profile: dict
    latency_ms: float
    reviewed: bool = False
    review_reasoning: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    model_threshold: float
    drift_status: dict
    uptime_sec: float