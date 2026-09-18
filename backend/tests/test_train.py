import numpy as np
import pytest
from tests.conftest import synthetic_transactions
from train import choose_threshold, evaluate, expected_cost, temporal_split


def test_temporal_split_has_no_future_leakage():
    frame = synthetic_transactions(2000)
    train, val, test, train_end, val_end = temporal_split(frame, 0.6, 0.2)

    assert train["step"].max() <= train_end < val["step"].min()
    assert val["step"].max() <= val_end < test["step"].min()
    assert len(train) + len(val) + len(test) == len(frame)


def test_expected_cost_counts_reviews_and_missed_fraud():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.9, 0.1, 0.9, 0.1])
    amounts = np.array([10.0, 10.0, 1000.0, 2000.0])

    # At 0.5: two alerts (rows 0 and 2) and one missed fraud worth 2000.
    assert expected_cost(y, scores, amounts, 0.5, 25.0) == 2 * 25.0 + 2000.0
    # Flag nothing: the whole fraud value is lost.
    assert expected_cost(y, scores, amounts, 1.1, 25.0) == 3000.0


def test_chosen_threshold_is_never_worse_than_the_default():
    """Cost at the chosen threshold should beat cost at 0.5."""
    rng = np.random.default_rng(3)
    y = (rng.random(4000) < 0.02).astype(int)
    scores = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 4000), 0, 1)
    amounts = np.where(y == 1, rng.uniform(1e4, 1e5, 4000), rng.uniform(10, 500, 4000))

    threshold, cost, curve = choose_threshold(y, scores, amounts, 25.0)
    assert 0 < threshold < 1
    assert cost <= expected_cost(y, scores, amounts, 0.5, 25.0)
    assert curve


def test_asymmetric_costs_push_the_threshold_down():
    """Cheap reviews and expensive fraud should push the threshold down."""
    rng = np.random.default_rng(7)
    y = (rng.random(3000) < 0.03).astype(int)
    scores = np.clip(y * 0.5 + rng.normal(0.25, 0.2, 3000), 0, 1)
    amounts = np.where(y == 1, 1e6, 100.0)

    cheap_reviews, _, _ = choose_threshold(y, scores, amounts, 1.0)
    dear_reviews, _, _ = choose_threshold(y, scores, amounts, 10_000.0)
    assert cheap_reviews < dear_reviews


def test_evaluate_reports_value_not_just_counts():
    y = np.array([0, 1, 1, 0])
    scores = np.array([0.1, 0.9, 0.2, 0.05])
    amounts = np.array([50.0, 1000.0, 400.0, 20.0])

    result = evaluate(y, scores, amounts, 0.5, 25.0)
    econ = result["economics"]
    assert econ["fraud_value_at_risk"] == 1400.0
    assert econ["fraud_value_caught"] == 1000.0
    assert econ["fraud_value_missed"] == 400.0
    assert result["confusion_matrix"]["true_positive"] == 1
    assert result["confusion_matrix"]["false_negative"] == 1


def test_split_cutoffs_follow_row_volume_not_step_values():
    """Step is skewed, so splitting on unique step values leaves too few test
    rows. Cutoffs come from row counts instead."""
    import numpy as np
    import pandas as pd
    from tests.conftest import synthetic_transactions

    frame = synthetic_transactions(4000)
    # Pile 90% of rows into the earliest hours, as the real file does.
    rng = np.random.default_rng(0)
    skewed = frame.copy()
    skewed["step"] = np.where(
        rng.random(len(frame)) < 0.9,
        rng.integers(1, 50, len(frame)),
        rng.integers(400, 744, len(frame)),
    )
    skewed = pd.DataFrame(skewed)

    train, val, test, _, _ = temporal_split(skewed, 0.6, 0.2)
    assert 0.4 < len(train) / len(skewed) < 0.8
    assert len(test) / len(skewed) > 0.1


def test_trivial_rule_baseline_recovers_the_paysim_identity():
    """Checks the rule picks up exactly the emptied-sender rows."""
    import pandas as pd
    from train import trivial_rule_baseline

    frame = pd.DataFrame(
        [
            # Fraud: sender emptied exactly.
            {"amount": 500.0, "oldbalanceorig": 500.0, "newbalanceorig": 0.0, "isfraud": 1},
            {"amount": 900.0, "oldbalanceorig": 900.0, "newbalanceorig": 0.0, "isfraud": 1},
            # Legit: balance drawn down normally.
            {"amount": 100.0, "oldbalanceorig": 900.0, "newbalanceorig": 800.0, "isfraud": 0},
            {"amount": 10.0, "oldbalanceorig": 50.0, "newbalanceorig": 40.0, "isfraud": 0},
        ]
    )
    result = trivial_rule_baseline(frame)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["alert_rate"] == 0.5


def test_fast_threshold_sweep_matches_brute_force():
    """The fast sweep should give the same answer as the slow one."""
    rng = np.random.default_rng(11)
    n = 5000
    y = (rng.random(n) < 0.02).astype(int)
    scores = np.clip(y * 0.5 + rng.normal(0.25, 0.2, n), 0, 1)
    amounts = np.where(y == 1, rng.uniform(1e4, 1e5, n), rng.uniform(10, 500, n))
    review_cost = 25.0

    threshold, cost, _ = choose_threshold(y, scores, amounts, review_cost)

    candidates = np.unique(np.round(np.linspace(0.001, 0.999, 999), 4))
    brute = np.array([expected_cost(y, scores, amounts, t, review_cost) for t in candidates])
    assert threshold == pytest.approx(float(candidates[int(np.argmin(brute))]))
    assert cost == pytest.approx(float(brute.min()))
