import numpy as np
import pandas as pd
import pytest
from features import FEATURE_NAMES, TYPE_CODES, build_feature_row, build_features
from tests.conftest import synthetic_transactions


def test_feature_order_is_stable():
    frame = synthetic_transactions(50)
    assert list(build_features(frame).columns) == FEATURE_NAMES


def test_step_becomes_hour_of_day():
    frame = pd.DataFrame(
        [
            {
                "step": s,
                "types": 1,
                "amount": 1.0,
                "oldbalanceorig": 1.0,
                "newbalanceorig": 0.0,
                "oldbalancedest": 0.0,
                "newbalancedest": 0.0,
                "isflaggedfraud": 0.0,
            }
            for s in (0, 23, 24, 49, 744)
        ]
    )
    assert build_features(frame)["hour_of_day"].tolist() == [0, 23, 0, 1, 0]


def test_ledger_residuals():
    """A transfer that adds up leaves nothing left over on the sender side."""
    frame = pd.DataFrame(
        [
            {
                "step": 1,
                "types": 4,
                "amount": 100.0,
                "oldbalanceorig": 500.0,
                "newbalanceorig": 400.0,
                "oldbalancedest": 200.0,
                "newbalancedest": 300.0,
                "isflaggedfraud": 0.0,
            }
        ]
    )
    row = build_features(frame).iloc[0]
    assert row["error_balance_orig"] == pytest.approx(0.0)
    assert row["error_balance_dest"] == pytest.approx(0.0)


def test_emptied_account_flag():
    frame = pd.DataFrame(
        [
            {
                "step": 1,
                "types": 4,
                "amount": 500.0,
                "oldbalanceorig": 500.0,
                "newbalanceorig": 0.0,
                "oldbalancedest": 0.0,
                "newbalancedest": 0.0,
                "isflaggedfraud": 0.0,
            },
            {
                "step": 1,
                "types": 4,
                "amount": 100.0,
                "oldbalanceorig": 500.0,
                "newbalanceorig": 400.0,
                "oldbalancedest": 0.0,
                "newbalancedest": 0.0,
                "isflaggedfraud": 0.0,
            },
        ]
    )
    assert build_features(frame)["orig_emptied"].tolist() == [1, 0]


def test_zero_balance_does_not_divide_by_zero():
    frame = pd.DataFrame(
        [
            {
                "step": 1,
                "types": 1,
                "amount": 900.0,
                "oldbalanceorig": 0.0,
                "newbalanceorig": 0.0,
                "oldbalancedest": 0.0,
                "newbalancedest": 0.0,
                "isflaggedfraud": 0.0,
            }
        ]
    )
    assert np.isfinite(build_features(frame)["amount_to_balance_ratio"]).all()


def test_single_row_path_matches_frame_path():
    """Single row and batch paths need to produce the same thing."""
    payload = {
        "step": 33,
        "types": 2,
        "amount": 4200.0,
        "oldbalanceorig": 9000.0,
        "newbalanceorig": 4800.0,
        "oldbalancedest": 100.0,
        "newbalancedest": 4300.0,
        "isflaggedfraud": 0.0,
    }
    np.testing.assert_allclose(
        build_feature_row(payload),
        build_features(pd.DataFrame([payload])).to_numpy(dtype="float64"),
    )


def test_type_codes_match_alphabetical_label_encoding():
    assert TYPE_CODES == {name: index for index, name in enumerate(sorted(TYPE_CODES))}


def test_realistic_feature_set_excludes_the_ledger_identity():
    """The features that give away the simulator are listed separately so they
    can be dropped."""
    from features import DEGENERATE_FEATURES, FEATURE_SETS

    assert set(FEATURE_SETS["full"]) == set(FEATURE_NAMES)
    assert not set(FEATURE_SETS["realistic"]) & set(DEGENERATE_FEATURES)
    assert len(FEATURE_SETS["realistic"]) == len(FEATURE_NAMES) - len(DEGENERATE_FEATURES)


def test_build_features_honours_a_column_subset():
    from features import FEATURE_SETS

    frame = synthetic_transactions(20)
    subset = build_features(frame, FEATURE_SETS["realistic"])
    assert list(subset.columns) == FEATURE_SETS["realistic"]
