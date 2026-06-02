"""
Pydantic models for /score request and response.
Defines the contract between the client and the API.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class TransactionRequest(BaseModel):
    card_id: str = Field(..., example="card_0042")
    merchant_id: str = Field(..., example="merch_017")
    amount: float = Field(..., gt=0, example=12400.00)
    merchant_category: str = Field(..., example="electronics")
    timestamp: datetime = Field(..., example="2025-05-19T03:22:11")

    class Config:
        json_schema_extra = {
            "example": {
                "card_id": "card_0042",
                "merchant_id": "merch_017",
                "amount": 12400.00,
                "merchant_category": "electronics",
                "timestamp": "2025-05-19T03:22:11"
            }
        }


class VelocityFeatures(BaseModel):
    txn_count_1hr: int
    txn_count_24hr: int
    amount_mean_7d: float
    amount_zscore: float
    unique_merchants_1hr: int
    time_since_last_txn_sec: Optional[float]


class BehavioralProfile(BaseModel):
    avg_amount: float
    top_hour: Optional[int]
    top_category: Optional[str]
    total_txns_seen: int


class ScoreResponse(BaseModel):
    card_id: str
    fraud_probability: float
    decision: str                        # BLOCK or ALLOW
    threshold_used: float
    top_signals: list[str]
    velocity: VelocityFeatures
    behavioral_profile: BehavioralProfile
    latency_ms: float
