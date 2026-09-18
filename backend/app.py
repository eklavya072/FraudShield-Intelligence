"""FraudShield API.

Serves the model, the review queue, batch scoring and the drift check.

If SHAP can't produce an attribution this returns available: false rather than
a made up number. You can't tell a fake attribution from a real one, and you'd
act on it either way.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from drift import compute as compute_drift
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from features import DISPLAY_NAMES, FEATURE_NAMES, build_feature_row, build_features
from pydantic import BaseModel, Field, field_validator
from store import VALID_DECISIONS, AlertStore

try:
    import pandas as pd
except ImportError:  # pragma: no cover - pandas is a hard dependency of features
    pd = None

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
API_KEY = os.getenv("FRAUDSHIELD_API_KEY")  # unset => open (fine for a local demo)
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "5000"))

state: dict = {
    "model": None,
    "threshold": 0.5,
    "explainer": None,
    "explainer_error": None,
    "metrics": {},
    "profile": {},
    "store": None,
    "trained_at": None,
    "feature_names": FEATURE_NAMES,
}


def _load_artifacts() -> None:
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

    for key, filename in (("metrics", "metrics.json"), ("profile", "reference_profile.json")):
        path = MODEL_DIR / filename
        state[key] = json.loads(path.read_text()) if path.exists() else {}

    if shap is None:
        state["explainer_error"] = "shap is not installed"
    else:
        try:
            state["explainer"] = shap.TreeExplainer(state["model"])
        except Exception as exc:  # pragma: no cover - depends on shap/xgboost build
            state["explainer_error"] = f"{type(exc).__name__}: {exc}"
            logger.warning("SHAP explainer unavailable: %s", state["explainer_error"])

    state["store"] = AlertStore()
    logger.info("Loaded model trained %s, threshold %.4f", state["trained_at"], state["threshold"])


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_artifacts()
    yield
    if state["store"] is not None:
        state["store"].close()


app = FastAPI(
    title="FraudShield Intelligence API",
    description=(
        "Cost-thresholded fraud scoring, analyst triage ranked by expected loss, "
        "batch scoring and drift monitoring."
    ),
    version="3.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
# Auth and rate limiting
# --------------------------------------------------------------------------

_hits: dict[str, deque] = defaultdict(deque)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    """Sliding window limiter. In memory, so it only works with one worker."""
    if request.url.path in {"/", "/health", "/docs", "/openapi.json"}:
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


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------


class Transaction(BaseModel):
    step: int = Field(..., ge=0, description="Hour index of the transaction")
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
        "amount", "oldbalanceorig", "newbalanceorig", "oldbalancedest", "newbalancedest"
    )
    @classmethod
    def finite(cls, value: float) -> float:
        if not np.isfinite(value):
            raise ValueError("value must be finite")
        return value


class BatchRequest(BaseModel):
    transactions: list[Transaction] = Field(..., min_length=1)


class Decision(BaseModel):
    decision: Literal["confirmed_fraud", "false_positive"]
    note: str | None = Field(default=None, max_length=1000)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def explain(feature_row: np.ndarray) -> dict:
    """Top SHAP values, or available: false with a reason. Check `available` first."""
    explainer = state["explainer"]
    if explainer is None:
        return {
            "available": False,
            "method": None,
            "reason": state["explainer_error"] or "explainer not initialised",
            "factors": [],
        }

    try:
        values = np.array(explainer.shap_values(feature_row))
        if values.ndim == 3:  # (samples, features, classes)
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
        return {
            "available": True,
            "method": "shap.TreeExplainer",
            "reason": None,
            "factors": factors[:5],
        }
    except Exception as exc:
        logger.warning("SHAP failed for a request: %s", exc)
        return {
            "available": False,
            "method": None,
            "reason": f"shap failed: {type(exc).__name__}",
            "factors": [],
        }


def score_one(transaction: Transaction, source: str, explain_it: bool = True) -> dict:
    started = time.perf_counter()
    payload = transaction.model_dump()
    feature_row = build_feature_row(payload, state["feature_names"])

    probability = float(state["model"].predict_proba(feature_row)[0][1])
    threshold = state["threshold"]
    alerted = probability >= threshold
    expected_loss = probability * transaction.amount

    alert_id = state["store"].record(
        features=dict(zip(state["feature_names"], feature_row[0].tolist(), strict=True)),
        amount=transaction.amount,
        score=probability,
        threshold=threshold,
        source=source,
    )

    result = {
        "alert_id": alert_id,
        "decision": "ALERT" if alerted else "CLEAR",
        "fraud_probability": round(probability * 100, 2),
        "threshold": round(threshold * 100, 2),
        "expected_loss": round(expected_loss, 2),
        "model": "XGBoost",
        "model_trained_at": state["trained_at"],
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }
    if explain_it:
        result["explanation"] = explain(feature_row)
    return result


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@app.get("/", response_class=PlainTextResponse, include_in_schema=False)
async def root() -> str:
    return (
        "FraudShield Intelligence API\n"
        f"Model trained: {state['trained_at']}\n"
        f"Decision threshold: {state['threshold']:.4f}\n"
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


@app.get("/metrics")
async def metrics() -> dict:
    """Everything from the last training run: split, metrics, threshold, costs."""
    if not state["metrics"]:
        raise HTTPException(status_code=404, detail="metrics.json not found")
    return state["metrics"]


@app.post("/predict", dependencies=[Depends(require_api_key)])
async def predict(transaction: Transaction) -> dict:
    return score_one(transaction, source="api")


@app.post("/predict/batch", dependencies=[Depends(require_api_key)])
async def predict_batch(request: BatchRequest) -> dict:
    """Scores a whole file in one pass."""
    if len(request.transactions) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Batch of {len(request.transactions)} exceeds limit {MAX_BATCH_SIZE}",
        )

    started = time.perf_counter()
    frame = pd.DataFrame([t.model_dump() for t in request.transactions])
    matrix = build_features(frame, state["feature_names"])
    probabilities = state["model"].predict_proba(matrix)[:, 1]
    threshold = state["threshold"]
    amounts = frame["amount"].to_numpy()

    store = state["store"]
    results = []
    for index, probability in enumerate(probabilities):
        alert_id = store.record(
            features=matrix.iloc[index].to_dict(),
            amount=float(amounts[index]),
            score=float(probability),
            threshold=threshold,
            source="batch",
        )
        results.append(
            {
                "row": index,
                "alert_id": alert_id,
                "decision": "ALERT" if probability >= threshold else "CLEAR",
                "fraud_probability": round(float(probability) * 100, 2),
                "expected_loss": round(float(probability) * float(amounts[index]), 2),
            }
        )

    alerts = sum(1 for r in results if r["decision"] == "ALERT")
    return {
        "scored": len(results),
        "alerts": alerts,
        "alert_rate": round(alerts / len(results), 4),
        "flagged_exposure": round(
            sum(r["expected_loss"] for r in results if r["decision"] == "ALERT"), 2
        ),
        "threshold": round(threshold * 100, 2),
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "results": results,
    }


@app.get("/alerts", dependencies=[Depends(require_api_key)])
async def alerts(status: str = "pending", limit: int = 100) -> dict:
    if status != "all" and status not in VALID_DECISIONS | {"pending"}:
        raise HTTPException(status_code=400, detail=f"Unknown status '{status}'")
    return {
        "status": status,
        "alerts": state["store"].queue(status=status, limit=min(limit, 500)),
        "summary": state["store"].summary(),
    }


@app.post("/alerts/{alert_id}/decision", dependencies=[Depends(require_api_key)])
async def decide(alert_id: int, decision: Decision) -> dict:
    updated = state["store"].decide(alert_id, decision.decision, decision.note)
    if not updated:
        raise HTTPException(status_code=404, detail=f"No open alert with id {alert_id}")
    return {"alert_id": alert_id, "status": decision.decision}


@app.get("/drift", dependencies=[Depends(require_api_key)])
async def drift(limit: int = 5000) -> dict:
    if not state["profile"]:
        raise HTTPException(status_code=404, detail="reference_profile.json not found")
    return compute_drift(state["profile"], state["store"].recent_features(limit))
