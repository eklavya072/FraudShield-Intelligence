"""Client for the backend API.

Endpoint comes from API_URL only. The earlier version read API_URL and then
hardcoded the deployed URL at the call site, so running locally looked fine
while every request was going to production.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = float(os.getenv("API_TIMEOUT_SECONDS", "30"))


class ApiError(RuntimeError):
    """Raised when the backend is unreachable or returns an error."""


def base_url() -> str:
    return os.getenv("API_URL", DEFAULT_BASE_URL).rstrip("/")


def _headers() -> dict:
    key = os.getenv("FRAUDSHIELD_API_KEY")
    return {"X-API-Key": key} if key else {}


def _request(method: str, path: str, **kwargs):
    url = f"{base_url()}{path}"
    try:
        response = requests.request(method, url, headers=_headers(), timeout=TIMEOUT, **kwargs)
    except requests.exceptions.ConnectionError as exc:
        raise ApiError(f"Cannot reach the API at {url}. Is the backend running?") from exc
    except requests.exceptions.Timeout as exc:
        raise ApiError(f"The API did not respond within {TIMEOUT:.0f}s.") from exc

    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ApiError(f"{response.status_code} from {path}: {detail}")

    return response.json()


def health() -> dict:
    return _request("GET", "/health")


def predict(payload: dict) -> dict:
    return _request("POST", "/predict", json=payload)


def predict_batch(transactions: list[dict]) -> dict:
    return _request("POST", "/predict/batch", json={"transactions": transactions})


def alerts(status: str = "pending", limit: int = 100) -> dict:
    return _request("GET", "/alerts", params={"status": status, "limit": limit})


def decide(alert_id: int, decision: str, note: str | None = None) -> dict:
    return _request(
        "POST", f"/alerts/{alert_id}/decision", json={"decision": decision, "note": note}
    )


def drift(limit: int = 5000) -> dict:
    return _request("GET", "/drift", params={"limit": limit})


@st.cache_data(ttl=300, show_spinner=False)
def metrics() -> dict:
    """Cached, since this only changes when the model is retrained."""
    return _request("GET", "/metrics")
