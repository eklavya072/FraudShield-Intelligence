import pytest
from store import AlertStore


@pytest.fixture
def store(tmp_path):
    instance = AlertStore(tmp_path / "s.db")
    yield instance
    instance.close()


def test_only_scores_above_threshold_become_alerts(store):
    store.record({"amount": 10.0}, 10.0, score=0.9, threshold=0.5)
    store.record({"amount": 10.0}, 10.0, score=0.1, threshold=0.5)

    assert len(store.queue()) == 1
    assert store.summary()["scored"] == 2
    assert store.summary()["alerts"] == 1


def test_queue_orders_by_expected_loss(store):
    store.record({"amount": 100.0}, 100.0, score=0.99, threshold=0.5)  # 99
    store.record({"amount": 50_000.0}, 50_000.0, score=0.60, threshold=0.5)  # 30k

    queue = store.queue()
    assert [round(a["expected_loss"]) for a in queue] == [30_000, 99]


def test_decision_is_recorded_and_removes_from_pending(store):
    alert_id = store.record({"amount": 100.0}, 100.0, score=0.9, threshold=0.5)

    assert store.decide(alert_id, "confirmed_fraud", "verified") is True
    assert store.queue(status="pending") == []

    reviewed = store.queue(status="confirmed_fraud")
    assert reviewed[0]["analyst_note"] == "verified"
    assert reviewed[0]["decided_at"] is not None


def test_unknown_decision_rejected(store):
    alert_id = store.record({"amount": 1.0}, 1.0, score=0.9, threshold=0.5)
    with pytest.raises(ValueError):
        store.decide(alert_id, "maybe")


def test_realised_precision_is_none_before_any_review(store):
    store.record({"amount": 1.0}, 1.0, score=0.9, threshold=0.5)
    assert store.summary()["realised_precision"] is None


def test_realised_precision_tracks_analyst_outcomes(store):
    ids = [store.record({"amount": 1.0}, 1.0, 0.9, 0.5) for _ in range(4)]
    store.decide(ids[0], "confirmed_fraud")
    store.decide(ids[1], "confirmed_fraud")
    store.decide(ids[2], "false_positive")

    summary = store.summary()
    assert summary["reviewed"] == 3
    assert summary["realised_precision"] == pytest.approx(2 / 3)
    assert summary["pending"] == 1


def test_state_survives_reopening(tmp_path):
    first = AlertStore(tmp_path / "persist.db")
    alert_id = first.record({"amount": 5.0}, 5.0, 0.9, 0.5)
    first.decide(alert_id, "confirmed_fraud", "kept")
    first.close()

    second = AlertStore(tmp_path / "persist.db")
    assert second.summary()["confirmed"] == 1
    second.close()
