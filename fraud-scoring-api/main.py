import logging, time
from fastapi import FastAPI, Request, Response
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from app.routers.score import router
import os

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

# ── Security headers middleware ───────────────────────────────────────────────
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"]  = "nosniff"
        response.headers["X-Frame-Options"]          = "DENY"
        response.headers["X-XSS-Protection"]         = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Cache-Control"]             = "no-store"
        response.headers["Referrer-Policy"]           = "no-referrer"
        return response

app.add_middleware(SecurityHeadersMiddleware)

# ── Force HTTPS in production (set ENV=production to enable) ─────────────────
if os.getenv("ENV") == "production":
    app.add_middleware(HTTPSRedirectMiddleware)

app.include_router(router, tags=["Fraud Scoring"])

@app.get("/")
def root():
    return {"service": "fraud-scoring-api", "status": "running", "docs": "/docs"}