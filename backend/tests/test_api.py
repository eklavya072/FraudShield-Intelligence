def test_health_reports_model_and_explainer_state(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["threshold"] == 0.3
    assert isinstance(body["explanations_available"], bool)


def test_predict_returns_probability_and_decision(client, fraud_payload):
    body = client.post("/predict", json=fraud_payload).json()
    assert body["decision"] in {"ALERT", "CLEAR"}
    assert 0 <= body["fraud_probability"] <= 100
    assert body["expected_loss"] >= 0
    assert body["latency_ms"] >= 0


def test_decision_follows_the_threshold(client, fraud_payload):
    body = client.post("/predict", json=fraud_payload).json()
    over = body["fraud_probability"] >= body["threshold"]
    assert body["decision"] == ("ALERT" if over else "CLEAR")


def test_fraud_pattern_scores_above_a_benign_one(client, fraud_payload, legit_payload):
    fraud = client.post("/predict", json=fraud_payload).json()
    legit = client.post("/predict", json=legit_payload).json()
    assert fraud["fraud_probability"] > legit["fraud_probability"]


def test_explanations_are_never_fabricated(client, fraud_payload):
    """Real SHAP values, or available: false and an empty list.

    The old version made up attribution numbers from hardcoded constants when
    SHAP was missing and returned them in the same format as the real ones.
    """
    explanation = client.post("/predict", json=fraud_payload).json()["explanation"]

    if explanation["available"]:
        assert explanation["factors"]
        assert explanation["reason"] is None
        assert all("impact" in f and "direction" in f for f in explanation["factors"])
    else:
        assert explanation["factors"] == []
        assert explanation["reason"]


def test_at_most_five_factors_returned(client, fraud_payload):
    explanation = client.post("/predict", json=fraud_payload).json()["explanation"]
    assert len(explanation["factors"]) <= 5


def test_rejects_out_of_range_transaction_type(client, legit_payload):
    assert client.post("/predict", json={**legit_payload, "types": 9}).status_code == 422


def test_rejects_negative_amount(client, legit_payload):
    assert client.post("/predict", json={**legit_payload, "amount": -1}).status_code == 422


def test_rejects_missing_field(client, legit_payload):
    payload = {k: v for k, v in legit_payload.items() if k != "amount"}
    assert client.post("/predict", json=payload).status_code == 422


def test_rate_limit_returns_429(model_dir, tmp_path, monkeypatch):
    import sys

    from fastapi.testclient import TestClient

    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "3")
    sys.modules.pop("app", None)
    import app as app_module

    payload = {
        "step": 1,
        "types": 3,
        "amount": 1.0,
        "oldbalanceorig": 1.0,
        "newbalanceorig": 0.0,
        "oldbalancedest": 0.0,
        "newbalancedest": 0.0,
        "isflaggedfraud": 0.0,
    }
    with TestClient(app_module.app) as limited:
        codes = [limited.post("/predict", json=payload).status_code for _ in range(5)]
        assert 429 in codes
        # The limiter must not block anything else.
        assert limited.get("/health").status_code == 200
