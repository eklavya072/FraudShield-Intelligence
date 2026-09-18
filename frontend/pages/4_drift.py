"""Drift monitor.

Accuracy measured once at training time goes stale. This compares what the
model is seeing now against what it was trained on, per feature, using PSI.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from fraudshield import api, ui

ui.topbar("Drift monitor")

st.markdown(
    '<div class="page-title"><div><h1>Population <span>Drift</span></h1>'
    "<p>Live traffic against the training distribution</p></div></div>",
    unsafe_allow_html=True,
)

if ui.connection_banner() is None:
    st.stop()

ui.note(
    "<strong>PSI</strong> compares each feature now against what it looked like in "
    "training. Under 0.1 is stable, 0.1 to 0.25 is some shift, over 0.25 is worth "
    "retraining. If the <em>score</em> distribution moves while the features look "
    "stable, something changed that the features aren't picking up."
)

try:
    report = api.drift()
except api.ApiError as exc:
    st.error(str(exc))
    st.stop()

if not report["sufficient_data"]:
    st.info(
        f"Only {report['n_observations']} transactions scored so far. PSI needs at "
        f"least {report['min_observations']} before it means anything, and saying "
        "'stable' off a handful of rows would be worse than saying nothing. Score a "
        "batch to fill this in."
    )
    st.stop()

overall = report["overall"]
score_psi = report["score_psi"]
ui.metric_cards(
    [
        {
            "label": "Observations",
            "value": f"{report['n_observations']:,}",
            "hint": "recent scored traffic",
        },
        {
            "label": "Worst feature PSI",
            "value": f"{overall['max_feature_psi']:.3f}",
            "hint": overall["band"],
            "tone": "good" if overall["band"] == "stable" else "bad",
        },
        {
            "label": "Score PSI",
            "value": f"{score_psi:.3f}" if score_psi is not None else "--",
            "hint": report.get("score_band") or "",
            "tone": "good" if report.get("score_band") == "stable" else "bad",
        },
    ]
)

if overall["band"] == "significant":
    st.error(
        "Significant drift. The numbers on the Model card page don't describe what "
        "this model is currently seeing. Retrain before trusting them."
    )
elif overall["band"] == "moderate":
    st.warning("Some drift. Worth watching, not yet a reason to retrain.")

table = pd.DataFrame(report["features"])
table["status"] = table["band"]
st.markdown('<div class="panel-title">Per-feature stability</div>', unsafe_allow_html=True)
st.dataframe(
    table[["feature", "psi", "status", "reference_mean", "live_mean"]],
    use_container_width=True,
    height=430,
    column_config={
        "psi": st.column_config.ProgressColumn(
            "PSI",
            min_value=0.0,
            max_value=max(0.5, float(table["psi"].max())),
            format="%.3f",
        ),
        "reference_mean": st.column_config.NumberColumn("Training mean", format="%.2f"),
        "live_mean": st.column_config.NumberColumn("Live mean", format="%.2f"),
    },
)
