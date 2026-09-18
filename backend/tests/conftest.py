"""Builds a small real model so the API tests hit the actual code path."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from features import FEATURE_NAMES, RAW_FIELDS, build_features  # noqa: E402
from train import build_reference_profile  # noqa: E402


def synthetic_transactions(n: int = 800, seed: int = 0) -> pd.DataFrame:
    """Fraud here looks like PaySim's: a transfer that empties the sender."""
    rng = np.random.default_rng(seed)
    is_fraud = rng.random(n) < 0.08

    amount = np.where(is_fraud, rng.uniform(1e5, 9e5, n), rng.uniform(10, 5e4, n))
    old_orig = np.where(is_fraud, amount, rng.uniform(1e3, 1e6, n))
    new_orig = np.where(is_fraud, 0.0, np.maximum(old_orig - amount, 0))

    return pd.DataFrame(
        {
            "step": rng.integers(1, 200, n),
            "types": np.where(is_fraud, 4, rng.integers(0, 4, n)),
            "amount": amount,
            "oldbalanceorig": old_orig,
            "newbalanceorig": new_orig,
            "oldbalancedest": rng.uniform(0, 1e5, n),
            "newbalancedest": rng.uniform(0, 1e5, n),
            "isflaggedfraud": np.zeros(n),
            "isfraud": is_fraud.astype(int),
        }
    )[RAW_FIELDS + ["isfraud"]]


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("model")
    frame = synthetic_transactions()
    X, y = build_features(frame), frame["isfraud"].to_numpy()

    model = XGBClassifier(n_estimators=30, max_depth=3, tree_method="hist", n_jobs=1)
    model.fit(X, y)

    joblib.dump(
        {
            "model": model,
            "threshold": 0.3,
            "feature_names": FEATURE_NAMES,
            "trained_at": "2026-01-01T00:00:00+00:00",
            "review_cost": 25.0,
        },
        directory / "fraud_model.joblib",
    )
    (directory / "metrics.json").write_text(json.dumps({"test": {"average_precision": 0.9}}))
    (directory / "reference_profile.json").write_text(
        json.dumps(build_reference_profile(X, model.predict_proba(X)[:, 1]))
    )
    return directory


@pytest.fixture
def client(model_dir, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    monkeypatch.setenv("FRAUDSHIELD_DB", str(tmp_path / "test.db"))
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "10000")

    for module in ("app", "store"):
        sys.modules.pop(module, None)
    import app as app_module

    with TestClient(app_module.app) as test_client:
        yield test_client


@pytest.fixture
def legit_payload() -> dict:
    return {
        "step": 10,
        "types": 3,
        "amount": 250.0,
        "oldbalanceorig": 50_000.0,
        "newbalanceorig": 49_750.0,
        "oldbalancedest": 1_000.0,
        "newbalancedest": 1_250.0,
        "isflaggedfraud": 0.0,
    }


@pytest.fixture
def fraud_payload() -> dict:
    return {
        "step": 10,
        "types": 4,
        "amount": 500_000.0,
        "oldbalanceorig": 500_000.0,
        "newbalanceorig": 0.0,
        "oldbalancedest": 0.0,
        "newbalancedest": 0.0,
        "isflaggedfraud": 0.0,
    }
