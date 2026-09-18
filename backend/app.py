"""FraudShield API.

One endpoint that matters: POST /predict takes a transaction and returns a
fraud probability, whether that clears the alert threshold, and the SHAP
values behind it.

If SHAP can't produce an attribution this returns available: false with a
reason rather than a made up number. You can't tell a fake attribution from a
real one, and you'd act on it either way.
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from features import DISPLAY_NAMES, FEATURE_NAMES, build_feature_row
from pydantic import BaseModel, Field, field_validator

try:
    import shap
except ImportError:
    shap = None

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("fraudshield")

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).parent / "model"))

# This is a public demo endpoint on a free tier, so cap how fast one caller can
# hit it. In memory, which means it only works with a single worker.
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))

state: dict = {
    "model": None,
    "threshold": 0.5,
    "feature_names": FEATURE_NAMES,
    "explainer": None,
    "explainer_error": None,
    "trained_at": None,
}


def _load_model() -> None:
    bundle_path = MODEL_DIR / "fraud_model.joblib"
    if not bundle_path.exists():
        raise RuntimeError(f"No model at {bundle_path}. Run `python train.py` first.")

    bundle = joblib.load(bundle_path)
    state["model"] = bundle["model"]
    state["threshold"] = float(bundle["threshold"])
    state["trained_at"] = bundle.get("trained_at")
    # The columns this model was fitted on. Serving a different set would
    # misalign everything.
    state["feature_names"] = list(bundle.get("feature_names") or FEATURE_NAMES)

    if shap is None:
        state["explainer_error"] = "shap is not installed"
    else:
        try:
            state["explainer"] = shap.TreeExplainer(state["model"])
        except Exception as exc:  # pragma: no cover - depends on the shap build
            state["explainer_error"] = f"{type(exc).__name__}: {exc}"
            logger.warning("SHAP unavailable: %s", state["explainer_error"])

    logger.info("Loaded model trained %s, threshold %.4f", state["trained_at"], state["threshold"])


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_model()
    yield


app = FastAPI(
    title="FraudShield API",
    description="Fraud scoring for PaySim transactions, with SHAP explanations.",
    version="3.0.0",
    lifespan=lifespan,
)

_hits: dict[str, deque] = defaultdict(deque)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.url.path != "/predict":
        return await call_next(request)

    client = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = _hits[client]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT:
        return JSONResponse(
            status_code=429,
            content={"detail": f"Rate limit of {RATE_LIMIT} requests/minute exceeded"},
        )
    window.append(now)
    return await call_next(request)


class Transaction(BaseModel):
    step: int = Field(..., ge=0, description="Hour of the transaction")
    types: int = Field(
        ...,
        ge=0,
        le=4,
        description="0 CASH_IN, 1 CASH_OUT, 2 DEBIT, 3 PAYMENT, 4 TRANSFER",
    )
    amount: float = Field(..., ge=0)
    oldbalanceorig: float = Field(..., ge=0)
    newbalanceorig: float = Field(..., ge=0)
    oldbalancedest: float = Field(..., ge=0)
    newbalancedest: float = Field(..., ge=0)
    isflaggedfraud: float = Field(0.0, ge=0, le=1)

    @field_validator(
        "amount",
        "oldbalanceorig",
        "newbalanceorig",
        "oldbalancedest",
        "newbalancedest",
    )
    @classmethod
    def finite(cls, value: float) -> float:
        if not np.isfinite(value):
            raise ValueError("value must be finite")
        return value


def explain(feature_row: np.ndarray) -> dict:
    """Top SHAP values, or available: false with a reason. Check `available` first."""
    explainer = state["explainer"]
    if explainer is None:
        return {
            "available": False,
            "reason": state["explainer_error"] or "explainer not initialised",
            "factors": [],
        }

    try:
        values = np.array(explainer.shap_values(feature_row))
        if values.ndim == 3:  # (rows, features, classes)
            values = values[:, :, -1]
        if values.ndim == 2:
            values = values[0]

        factors = [
            {
                "feature": DISPLAY_NAMES.get(name, name),
                "impact": round(float(value), 4),
                "direction": "increases risk" if value > 0 else "reduces risk",
            }
            for name, value in zip(state["feature_names"], values, strict=True)
        ]
        factors.sort(key=lambda item: abs(item["impact"]), reverse=True)
        return {"available": True, "reason": None, "factors": factors[:5]}
    except Exception as exc:
        logger.warning("SHAP failed on a request: %s", exc)
        return {
            "available": False,
            "reason": f"shap failed: {type(exc).__name__}",
            "factors": [],
        }


@app.get("/", response_class=PlainTextResponse, include_in_schema=False)
async def root() -> str:
    return (
        "FraudShield API\n"
        f"Model trained: {state['trained_at']}\n"
        f"Alert threshold: {state['threshold']:.4f}\n"
        "Docs: /docs\n"
    )


@app.get("/health")
async def health() -> dict:
    """Health check, plus what model is loaded and whether SHAP is working."""
    return {
        "status": "ok" if state["model"] is not None else "degraded",
        "model_loaded": state["model"] is not None,
        "model_trained_at": state["trained_at"],
        "threshold": state["threshold"],
        "explanations_available": state["explainer"] is not None,
        "explanations_unavailable_reason": state["explainer_error"],
    }


@app.post("/predict")
async def predict(transaction: Transaction) -> dict:
    started = time.perf_counter()
    feature_row = build_feature_row(transaction.model_dump(), state["feature_names"])

    probability = float(state["model"].predict_proba(feature_row)[0][1])
    threshold = state["threshold"]

    return {
        "decision": "ALERT" if probability >= threshold else "CLEAR",
        "fraud_probability": round(probability * 100, 2),
        "threshold": round(threshold * 100, 2),
        "expected_loss": round(probability * transaction.amount, 2),
        "model": "XGBoost",
        "model_trained_at": state["trained_at"],
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "explanation": explain(feature_row),
    }
