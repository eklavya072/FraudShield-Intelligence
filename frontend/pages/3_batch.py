"""Upload a CSV, get back a sorted worklist."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st
from fraudshield import api, ui
from fraudshield.batch import (
    UploadError,
    normalise,
    to_payloads,
)

ui.topbar("Batch scoring")

st.markdown(
    '<div class="page-title"><div><h1>Batch <span>Scoring</span></h1>'
    "<p>Score a file of transactions and export the alerts</p></div></div>",
    unsafe_allow_html=True,
)

if ui.connection_banner() is None:
    st.stop()

ui.note(
    "Upload a CSV with the PaySim columns. Kaggle's own column names work. Everything "
    "is scored in one call and anything above the cutoff goes into the "
    "<strong>review queue</strong>. The output is sorted by expected loss so it comes "
    "back as a worklist rather than just a pile of rows."
)

uploaded = st.file_uploader("Transaction CSV", type=["csv"])
if uploaded is None:
    st.stop()

try:
    frame = normalise(pd.read_csv(uploaded))
except UploadError as exc:
    st.error(str(exc))
    st.stop()

limit = st.number_input("Rows to score", 1, 5000, min(len(frame), 1000))
frame = frame.head(int(limit))
st.caption(f"{len(frame):,} rows ready.")

if not st.button("Score batch", use_container_width=True):
    st.stop()

transactions = to_payloads(frame)

with st.spinner("Scoring..."):
    try:
        response = api.predict_batch(transactions)
    except api.ApiError as exc:
        st.error(str(exc))
        st.stop()

ui.metric_cards(
    [
        {"label": "Scored", "value": f"{response['scored']:,}"},
        {
            "label": "Alerts",
            "value": f"{response['alerts']:,}",
            "hint": f"{response['alert_rate']:.2%} alert rate",
        },
        {
            "label": "Flagged exposure",
            "value": ui.money(response["flagged_exposure"]),
            "hint": "expected loss across alerts",
        },
        {
            "label": "Throughput",
            "value": f"{response['scored'] / max(response['latency_ms'] / 1000, 1e-6):,.0f}/s",
            "hint": f"{response['latency_ms']:.0f} ms total",
        },
    ]
)

results = pd.DataFrame(response["results"])
scored = (
    frame.reset_index(drop=True)
    .join(results.set_index("row")[["alert_id", "decision", "fraud_probability", "expected_loss"]])
    .sort_values("expected_loss", ascending=False)
)

st.dataframe(scored, use_container_width=True, height=420)

buffer = io.StringIO()
scored.to_csv(buffer, index=False)
st.download_button(
    "Download scored worklist (CSV)",
    buffer.getvalue(),
    file_name="fraudshield_scored.csv",
    mime="text/csv",
    use_container_width=True,
)

if "isFraud" in scored.columns or "isfraud" in scored.columns:
    truth_column = "isFraud" if "isFraud" in scored.columns else "isfraud"
    truth = scored[truth_column].astype(int)
    flagged = scored["decision"].eq("ALERT")
    caught = int((truth.eq(1) & flagged).sum())
    total_fraud = int(truth.sum())
    if total_fraud:
        st.success(
            f"This file had labels: caught {caught} of {total_fraud} frauds "
            f"({caught / total_fraud:.1%}) while flagging {flagged.sum():,} of "
            f"{len(scored):,} rows."
        )
