"""Turns an uploaded CSV into API payloads.

Split out of the page so it can be tested without starting Streamlit.
"""

from __future__ import annotations

import pandas as pd

REQUIRED = [
    "step",
    "types",
    "amount",
    "oldbalanceorig",
    "newbalanceorig",
    "oldbalancedest",
    "newbalancedest",
]

# Kaggle's column names to the API's.
ALIASES = {
    "type": "types",
    "oldbalanceOrg": "oldbalanceorig",
    "newbalanceOrig": "newbalanceorig",
    "oldbalanceDest": "oldbalancedest",
    "newbalanceDest": "newbalancedest",
    "isFlaggedFraud": "isflaggedfraud",
}

TYPE_CODES = {"CASH_IN": 0, "CASH_OUT": 1, "DEBIT": 2, "PAYMENT": 3, "TRANSFER": 4}


class UploadError(ValueError):
    """The uploaded file cannot be scored as-is."""


def normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Renames columns, encodes the type column, checks nothing is missing."""
    frame = frame.rename(columns=ALIASES)

    if "types" in frame.columns and not pd.api.types.is_numeric_dtype(frame["types"]):
        # Checking "not numeric" rather than dtype == object. pandas 3 gives
        # string columns their own dtype, so the object check skipped this
        # entirely and then every row failed to convert to float.
        mapped = frame["types"].map(TYPE_CODES)
        if mapped.isna().any():
            unknown = sorted(set(frame.loc[mapped.isna(), "types"].dropna().unique()))
            raise UploadError(
                f"Unrecognised transaction type(s): {', '.join(map(str, unknown))}. "
                f"Expected one of: {', '.join(TYPE_CODES)}."
            )
        frame["types"] = mapped

    if "isflaggedfraud" not in frame.columns:
        frame["isflaggedfraud"] = 0.0

    missing = [column for column in REQUIRED if column not in frame.columns]
    if missing:
        raise UploadError(f"Missing required column(s): {', '.join(missing)}")

    return frame.dropna(subset=REQUIRED).reset_index(drop=True)


def to_payloads(frame: pd.DataFrame) -> list[dict]:
    """Rows to dicts the API will accept."""
    records = frame[REQUIRED + ["isflaggedfraud"]].astype(float).to_dict("records")
    for record in records:
        record["step"] = int(record["step"])
        record["types"] = int(record["types"])
    return records
