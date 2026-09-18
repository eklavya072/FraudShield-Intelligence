"""Review queue.

A score isn't a decision. This is the page an analyst would actually use: open
alerts sorted by expected loss (probability x amount), so the expensive ones
come first instead of the most confident ones.
"""

from __future__ import annotations

import html

import streamlit as st
from fraudshield import api, ui

ui.topbar("Analyst triage")

st.markdown(
    '<div class="page-title"><div><h1>Review <span>Queue</span></h1>'
    "<p>Open alerts ranked by expected loss</p></div></div>",
    unsafe_allow_html=True,
)

if ui.connection_banner() is None:
    st.stop()

ui.note(
    "Sorted by <strong>expected loss</strong> (probability x amount) rather than by "
    "probability, so a 60% chance on $80,000 gets looked at before a 98% chance on "
    "$40. Decisions made here feed the <strong>realised precision</strong> number "
    "below, which is precision on alerts that were actually reviewed rather than on "
    "the test set."
)

status = st.radio(
    "Queue",
    ["pending", "confirmed_fraud", "false_positive", "all"],
    horizontal=True,
    format_func=lambda s: {
        "pending": "Pending review",
        "confirmed_fraud": "Confirmed fraud",
        "false_positive": "False positives",
        "all": "All alerts",
    }[s],
    label_visibility="collapsed",
)

try:
    payload = api.alerts(status=status, limit=100)
except api.ApiError as exc:
    st.error(str(exc))
    st.stop()

summary = payload["summary"]
realised = summary["realised_precision"]
ui.metric_cards(
    [
        {
            "label": "Scored",
            "value": f"{summary['scored']:,}",
            "hint": "transactions through the API",
        },
        {
            "label": "Alerts raised",
            "value": f"{summary['alerts']:,}",
            "hint": f"{summary['alerts'] / summary['scored']:.2%} of traffic"
            if summary["scored"]
            else "",
        },
        {
            "label": "Pending review",
            "value": f"{summary['pending']:,}",
            "hint": "awaiting an analyst",
        },
        {
            "label": "Exposure in queue",
            "value": ui.money(summary["pending_exposure"]),
            "hint": "sum of expected loss",
        },
        {
            "label": "Realised precision",
            "value": f"{realised:.1%}" if realised is not None else "--",
            "hint": f"over {summary['reviewed']} reviewed"
            if summary["reviewed"]
            else "no reviews yet",
            "tone": None if realised is None else ("good" if realised >= 0.5 else "bad"),
        },
    ]
)

alerts = payload["alerts"]
if not alerts:
    st.markdown(
        '<div class="panel"><div class="panel-title">Nothing in the queue</div>'
        '<div class="panel-subtitle">Score something on the Risk console, or upload '
        "a file on the Batch scoring page.</div></div>",
        unsafe_allow_html=True,
    )
    st.stop()

for alert in alerts:
    features = alert["features"]
    type_label = ui.TYPE_LABELS.get(int(features.get("types", -1)), "Unknown")
    decided = alert["status"] != "pending"

    st.markdown(
        f"""
        <div class="alert-card">
          <div class="alert-head">
            <span class="alert-loss">{ui.money(alert["expected_loss"])} expected loss</span>
            <span class="alert-meta">#{alert["id"]} &middot;
              {alert["score"] * 100:.1f}% fraud probability &middot;
              {html.escape(type_label)} &middot;
              {ui.money(alert["amount"])} &middot;
              {html.escape(alert["source"])}</span>
          </div>
          <div class="alert-meta" style="margin-top:0.4rem;">
            Sender {ui.money(features.get("oldbalanceorig", 0))} &rarr;
            {ui.money(features.get("newbalanceorig", 0))} &middot;
            ledger mismatch {ui.money(features.get("error_balance_orig", 0))} &middot;
            hour {int(features.get("hour_of_day", 0))}
            {"&middot; <strong>" + html.escape(alert["status"]) + "</strong>" if decided else ""}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if decided:
        if alert["analyst_note"]:
            st.caption(f"Note: {alert['analyst_note']}")
        continue

    with st.form(f"decide_{alert['id']}"):
        note_text = st.text_input(
            "Analyst note",
            key=f"note_{alert['id']}",
            placeholder="Optional context for the audit trail",
            label_visibility="collapsed",
        )
        confirm_col, dismiss_col = st.columns(2)
        confirmed = confirm_col.form_submit_button("Confirm fraud", use_container_width=True)
        dismissed = dismiss_col.form_submit_button("False positive", use_container_width=True)

    if confirmed or dismissed:
        decision = "confirmed_fraud" if confirmed else "false_positive"
        try:
            api.decide(alert["id"], decision, note_text or None)
        except api.ApiError as exc:
            st.error(str(exc))
        else:
            st.rerun()
