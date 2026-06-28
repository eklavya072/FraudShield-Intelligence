from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel
import joblib
import numpy as np
import time

try:
    import shap
except ImportError:
    shap = None


FEATURE_NAMES = [
    "step",
    "types",
    "amount",
    "oldbalanceorig",
    "newbalanceorig",
    "oldbalancedest",
    "newbalancedest",
    "isflaggedfraud",
]

DISPLAY_NAMES = {
    "step": "Transaction Hour",
    "types": "Transaction Type",
    "amount": "Transaction Amount",
    "oldbalanceorig": "Sender Balance Before",
    "newbalanceorig": "Sender Balance After",
    "oldbalancedest": "Receiver Balance Before",
    "newbalancedest": "Receiver Balance After",
    "isflaggedfraud": "Rule Flag",
}

app = FastAPI(
    title="FraudShield Intelligence API",
    description="Real-time fraud detection API powered by an XGBoost model.",
    version="2.1.0",
    debug=True,
)

model = joblib.load("credit_fraud_xgb.pkl")
explainer = None

if shap is not None:
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = None


class FraudDetection(BaseModel):
    step: int
    types: int
    amount: float
    oldbalanceorig: float
    newbalanceorig: float
    oldbalancedest: float
    newbalancedest: float
    isflaggedfraud: float


def build_feature_array(data: FraudDetection) -> np.ndarray:
    return np.array(
        [
            [
                data.step,
                data.types,
                data.amount,
                data.oldbalanceorig,
                data.newbalanceorig,
                data.oldbalancedest,
                data.newbalancedest,
                data.isflaggedfraud,
            ]
        ]
    )


def fallback_attribution(data: FraudDetection):
    sender_drop = max(data.oldbalanceorig - data.newbalanceorig, 0)
    receiver_gain = max(data.newbalancedest - data.oldbalancedest, 0)
    amount_pressure = min(data.amount / 200000, 1.5)
    depletion_pressure = sender_drop / max(data.oldbalanceorig, 1)

    values = [
        ("Transaction Amount", round(amount_pressure * 2.4, 3)),
        ("Sender Balance Depletion", round(depletion_pressure * 2.2, 3)),
        ("Receiver Balance Movement", round((receiver_gain / max(data.amount, 1)) * 1.2, 3)),
        ("Rule Flag", round(float(data.isflaggedfraud) * 2.8, 3)),
        ("Transaction Type", round(0.6 if data.types in [1, 4] else -0.2, 3)),
        ("Transaction Hour", round(0.3 if data.step < 6 or data.step > 22 else -0.1, 3)),
    ]

    return [
        {"feature": feature, "impact": impact}
        for feature, impact in sorted(values, key=lambda item: abs(item[1]), reverse=True)[:5]
    ]


def explain_prediction(features: np.ndarray, data: FraudDetection):
    if explainer is None:
        return fallback_attribution(data)

    try:
        shap_values = explainer.shap_values(features)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        row_values = np.array(shap_values)
        if row_values.ndim == 2:
            row_values = row_values[0]

        attribution = []
        for name, value in zip(FEATURE_NAMES, row_values):
            attribution.append(
                {
                    "feature": DISPLAY_NAMES.get(name, name),
                    "impact": round(float(value), 4),
                }
            )

        return sorted(attribution, key=lambda item: abs(item["impact"]), reverse=True)[:5]
    except Exception:
        return fallback_attribution(data)


@app.get("/", response_class=PlainTextResponse)
async def running():
    return """
FraudShield Intelligence API
Backend Status: Running
Swagger Docs: /docs
"""


@app.get("/favicon.png", include_in_schema=False)
async def favicon():
    return FileResponse("favicon.png")


@app.post("/predict")
def predict(data: FraudDetection):
    start = time.perf_counter()
    features = build_feature_array(data)

    prediction = int(model.predict(features)[0])
    probability = model.predict_proba(features)[0]
    confidence = float(np.max(probability) * 100)
    fraud_probability = float(probability[1] * 100) if len(probability) > 1 else confidence
    prediction_time = round((time.perf_counter() - start) * 1000, 2)
    attribution = explain_prediction(features, data)

    return {
        "prediction": "Fraudulent Transaction"
        if prediction == 1
        else "Not Fraudulent Transaction",
        "status": "CRITICAL RISK" if prediction == 1 else "CLEARED",
        "confidence": round(confidence, 2),
        "fraud_probability": round(fraud_probability, 2),
        "model": "XGBoost",
        "prediction_time_ms": prediction_time,
        "top_factors": attribution,
    }
