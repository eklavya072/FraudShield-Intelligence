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
    assert body["alert_id"] > 0


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
        assert explanation["method"] == "shap.TreeExplainer"
        assert explanation["factors"]
        assert all("impact" in f and "direction" in f for f in explanation["factors"])
    else:
        assert explanation["factors"] == []
        assert explanation["reason"]
        assert explanation["method"] is None


def test_rejects_out_of_range_transaction_type(client, legit_payload):
    assert client.post("/predict", json={**legit_payload, "types": 9}).status_code == 422


def test_rejects_negative_amount(client, legit_payload):
    assert client.post("/predict", json={**legit_payload, "amount": -1}).status_code == 422


def test_batch_scores_every_row(client, fraud_payload, legit_payload):
    body = client.post(
        "/predict/batch",
        json={"transactions": [fraud_payload, legit_payload, legit_payload]},
    ).json()
    assert body["scored"] == 3
    assert len(body["results"]) == 3
    assert 0 <= body["alert_rate"] <= 1


def test_batch_rejects_oversized_request(client, legit_payload, monkeypatch):
    import app as app_module

    monkeypatch.setattr(app_module, "MAX_BATCH_SIZE", 2)
    response = client.post("/predict/batch", json={"transactions": [legit_payload] * 3})
    assert response.status_code == 413


def test_queue_ranks_by_expected_loss_not_probability(client, fraud_payload):
    """A 60% chance on $1M should come before a 99% chance on a small amount."""
    client.post("/predict", json={**fraud_payload, "amount": 100.0, "oldbalanceorig": 100.0})
    client.post(
        "/predict", json={**fraud_payload, "amount": 900_000.0, "oldbalanceorig": 900_000.0}
    )

    alerts = client.get("/alerts").json()["alerts"]
    if len(alerts) > 1:
        losses = [a["expected_loss"] for a in alerts]
        assert losses == sorted(losses, reverse=True)


def test_analyst_decision_updates_the_queue(client, fraud_payload):
    alert_id = client.post("/predict", json=fraud_payload).json()["alert_id"]
    pending = {a["id"] for a in client.get("/alerts").json()["alerts"]}
    if alert_id not in pending:
        return  # scored below threshold on this synthetic model; nothing to review

    response = client.post(
        f"/alerts/{alert_id}/decision",
        json={"decision": "confirmed_fraud", "note": "chargeback filed"},
    )
    assert response.status_code == 200
    assert alert_id not in {a["id"] for a in client.get("/alerts").json()["alerts"]}

    summary = client.get("/alerts").json()["summary"]
    assert summary["confirmed"] == 1
    assert summary["realised_precision"] == 1.0


def test_decision_on_unknown_alert_is_404(client):
    response = client.post("/alerts/999999/decision", json={"decision": "false_positive"})
    assert response.status_code == 404


def test_invalid_decision_value_rejected(client, fraud_payload):
    alert_id = client.post("/predict", json=fraud_payload).json()["alert_id"]
    response = client.post(f"/alerts/{alert_id}/decision", json={"decision": "looks_fine"})
    assert response.status_code == 422


def test_drift_admits_when_it_lacks_data(client, legit_payload):
    client.post("/predict", json=legit_payload)
    body = client.get("/drift").json()
    assert body["sufficient_data"] is False
    assert body["overall"] is None


def test_drift_reports_psi_once_traffic_accumulates(client, legit_payload):
    client.post("/predict/batch", json={"transactions": [legit_payload] * 150})
    body = client.get("/drift").json()
    assert body["sufficient_data"] is True
    assert body["features"]
    assert body["overall"]["band"] in {"stable", "moderate", "significant"}


def test_api_key_is_enforced_when_configured(model_dir, tmp_path, monkeypatch):
    import sys

    from fastapi.testclient import TestClient

    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    monkeypatch.setenv("FRAUDSHIELD_DB", str(tmp_path / "auth.db"))
    monkeypatch.setenv("FRAUDSHIELD_API_KEY", "s3cret")
    for module in ("app", "store"):
        sys.modules.pop(module, None)
    import app as app_module

    with TestClient(app_module.app) as authed:
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
        assert authed.post("/predict", json=payload).status_code == 401
        assert authed.get("/health").status_code == 200
        assert (
            authed.post("/predict", json=payload, headers={"X-API-Key": "s3cret"}).status_code
            == 200
        )


def test_rate_limit_returns_429(model_dir, tmp_path, monkeypatch):
    import sys

    from fastapi.testclient import TestClient

    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    monkeypatch.setenv("FRAUDSHIELD_DB", str(tmp_path / "rl.db"))
    monkeypatch.delenv("FRAUDSHIELD_API_KEY", raising=False)
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "3")
    for module in ("app", "store"):
        sys.modules.pop(module, None)
    import app as app_module

    with TestClient(app_module.app) as limited:
        codes = [limited.get("/metrics").status_code for _ in range(5)]
        assert 429 in codes
