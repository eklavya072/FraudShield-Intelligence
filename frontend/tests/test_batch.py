import pandas as pd
import pytest
from fraudshield.batch import REQUIRED, UploadError, normalise, to_payloads

RAW_KAGGLE = pd.DataFrame(
    [
        {
            "step": 1,
            "type": "PAYMENT",
            "amount": 9839.64,
            "nameOrig": "C123",
            "oldbalanceOrg": 170136.0,
            "newbalanceOrig": 160296.36,
            "nameDest": "M197",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "isFraud": 0,
            "isFlaggedFraud": 0,
        },
        {
            "step": 1,
            "type": "TRANSFER",
            "amount": 181.0,
            "nameOrig": "C130",
            "oldbalanceOrg": 181.0,
            "newbalanceOrig": 0.0,
            "nameDest": "C553",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "isFraud": 1,
            "isFlaggedFraud": 0,
        },
    ]
)


def test_raw_kaggle_headers_are_accepted():
    frame = normalise(RAW_KAGGLE.copy())
    assert all(column in frame.columns for column in REQUIRED)


def test_text_transaction_types_are_encoded():
    """pandas 3 gives string columns a str dtype, not object.

    The old dtype == object check skipped the mapping, and then every row
    failed to convert to float.
    """
    frame = normalise(RAW_KAGGLE.copy())
    assert frame["types"].tolist() == [3, 4]  # PAYMENT, TRANSFER
    assert pd.api.types.is_numeric_dtype(frame["types"])


def test_already_encoded_types_pass_through():
    frame = RAW_KAGGLE.rename(columns={"type": "types"}).copy()
    frame["types"] = [3, 4]
    assert normalise(frame)["types"].tolist() == [3, 4]


def test_unknown_type_is_reported():
    frame = RAW_KAGGLE.copy()
    frame.loc[0, "type"] = "WIRE"
    with pytest.raises(UploadError, match="WIRE"):
        normalise(frame)


def test_missing_column_is_reported():
    frame = RAW_KAGGLE.drop(columns=["amount"])
    with pytest.raises(UploadError, match="amount"):
        normalise(frame)


def test_missing_rule_flag_column_defaults_to_zero():
    frame = normalise(RAW_KAGGLE.drop(columns=["isFlaggedFraud"]))
    assert frame["isflaggedfraud"].tolist() == [0.0, 0.0]


def test_payloads_are_json_ready_with_correct_types():
    payloads = to_payloads(normalise(RAW_KAGGLE.copy()))
    assert len(payloads) == 2
    first = payloads[0]
    assert isinstance(first["step"], int)
    assert isinstance(first["types"], int)
    assert isinstance(first["amount"], float)
    assert set(first) == set(REQUIRED) | {"isflaggedfraud"}


def test_rows_with_missing_values_are_dropped():
    frame = RAW_KAGGLE.copy()
    frame.loc[0, "amount"] = None
    assert len(normalise(frame)) == 1
