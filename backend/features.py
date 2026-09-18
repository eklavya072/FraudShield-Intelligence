"""Feature building. Training and serving both import this so they can't drift apart."""

from __future__ import annotations

import numpy as np
import pandas as pd

# The eight fields a caller sends. Matches the raw PaySim columns.
RAW_FIELDS = [
    "step",
    "types",
    "amount",
    "oldbalanceorig",
    "newbalanceorig",
    "oldbalancedest",
    "newbalancedest",
    "isflaggedfraud",
]

# What the model sees. All derived from RAW_FIELDS, so adding features here
# doesn't change the request format.
FEATURE_NAMES = [
    "hour_of_day",
    "types",
    "amount",
    "oldbalanceorig",
    "newbalanceorig",
    "oldbalancedest",
    "newbalancedest",
    "isflaggedfraud",
    "error_balance_orig",
    "error_balance_dest",
    "orig_emptied",
    "amount_to_balance_ratio",
    "dest_was_empty",
]

DISPLAY_NAMES = {
    "hour_of_day": "Hour of Day",
    "types": "Transaction Type",
    "amount": "Transaction Amount",
    "oldbalanceorig": "Sender Balance Before",
    "newbalanceorig": "Sender Balance After",
    "oldbalancedest": "Receiver Balance Before",
    "newbalancedest": "Receiver Balance After",
    "isflaggedfraud": "Rule Flag",
    "error_balance_orig": "Sender Ledger Mismatch",
    "error_balance_dest": "Receiver Ledger Mismatch",
    "orig_emptied": "Sender Account Emptied",
    "amount_to_balance_ratio": "Amount vs Sender Balance",
    "dest_was_empty": "Receiver Started Empty",
}

# LabelEncoder sorts alphabetically. Pinned so re-encoding later can't shift these.
TYPE_CODES = {
    "CASH_IN": 0,
    "CASH_OUT": 1,
    "DEBIT": 2,
    "PAYMENT": 3,
    "TRANSFER": 4,
}


# 97.7% of PaySim frauds have oldbalanceOrg == amount and newbalanceOrig == 0,
# and 0% of legit rows do. Features built on that split the classes almost
# perfectly, but they're reading how the simulator generates fraud, not a real
# pattern. Kept because they're the best signal available here, listed
# separately so train.py can refit without them and show the difference.
DEGENERATE_FEATURES = [
    "error_balance_orig",
    "orig_emptied",
    "amount_to_balance_ratio",
]

REALISTIC_FEATURE_NAMES = [name for name in FEATURE_NAMES if name not in DEGENERATE_FEATURES]

FEATURE_SETS = {
    "full": FEATURE_NAMES,
    "realistic": REALISTIC_FEATURE_NAMES,
}


def build_features(raw: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """RAW_FIELDS in, feature matrix out. `columns` picks a subset, default is all."""
    out = pd.DataFrame(index=raw.index)

    # step is an hour counter over 30 days. Passing it raw breaks under a time
    # based split, since every test value is outside the training range. Hour of
    # day is the part that carries over.
    out["hour_of_day"] = raw["step"].astype("int64") % 24

    out["types"] = raw["types"].astype("int64")
    for col in (
        "amount",
        "oldbalanceorig",
        "newbalanceorig",
        "oldbalancedest",
        "newbalancedest",
        "isflaggedfraud",
    ):
        out[col] = raw[col].astype("float64")

    # Balances should satisfy new = old - amount for the sender and
    # new = old + amount for the receiver. Fraud rows often don't, so the
    # leftover is more useful than the raw balances.
    out["error_balance_orig"] = out["newbalanceorig"] + out["amount"] - out["oldbalanceorig"]
    out["error_balance_dest"] = out["oldbalancedest"] + out["amount"] - out["newbalancedest"]

    out["orig_emptied"] = ((out["newbalanceorig"] == 0) & (out["oldbalanceorig"] > 0)).astype(
        "int64"
    )
    out["amount_to_balance_ratio"] = out["amount"] / (out["oldbalanceorig"] + 1.0)
    out["dest_was_empty"] = (out["oldbalancedest"] == 0).astype("int64")

    return out[columns or FEATURE_NAMES]


def build_feature_row(payload: dict, columns: list[str] | None = None) -> np.ndarray:
    """Single-transaction path used by the API."""
    frame = pd.DataFrame([{field: payload[field] for field in RAW_FIELDS}])
    return build_features(frame, columns).to_numpy(dtype="float64")


def load_raw_csv(path: str, nrows: int | None = None) -> pd.DataFrame:
    """Read the original Kaggle PaySim file into RAW_FIELDS + isfraud."""
    frame = pd.read_csv(path, nrows=nrows)
    frame = frame.rename(
        columns={
            "type": "types",
            "oldbalanceOrg": "oldbalanceorig",
            "newbalanceOrig": "newbalanceorig",
            "oldbalanceDest": "oldbalancedest",
            "newbalanceDest": "newbalancedest",
            "isFlaggedFraud": "isflaggedfraud",
            "isFraud": "isfraud",
        }
    )
    frame["types"] = frame["types"].map(TYPE_CODES).astype("int64")
    return frame[RAW_FIELDS + ["isfraud"]]
