"""Scores one transaction and shows what drove the score."""

from __future__ import annotations

import html
import uuid

import streamlit as st
from fraudshield import api, ui

ui.topbar("Risk console")

st.markdown(
    '<div class="page-title"><div><h1>Transaction <span>Risk Console</span></h1>'
    "<p>Score one transaction against the cost-optimised threshold</p></div></div>",
    unsafe_allow_html=True,
)

health = ui.connection_banner()
if health is None:
    st.stop()

threshold_pct = health["threshold"] * 100
ui.note(
    f"Alerts fire at <strong>{threshold_pct:.2f}%</strong>, not 50%. The cutoff was "
    "picked by cost: missing a fraud costs the transaction amount, a false alarm costs "
    "one review. The <strong>Model card</strong> page has the working."
)

left, right = st.columns([0.42, 0.58], gap="large")

with left:
    with st.form("risk_form"):
        st.markdown('<div class="panel-title">Input parameters</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="field-label">Transaction hour (0-743)</div>', unsafe_allow_html=True
        )
        step = st.number_input(
            "step", min_value=0, max_value=743, value=10, label_visibility="collapsed"
        )

        st.markdown('<div class="field-label">Transaction type</div>', unsafe_allow_html=True)
        type_label = st.selectbox(
            "types", list(ui.TRANSACTION_TYPES), index=4, label_visibility="collapsed"
        )

        st.markdown('<div class="field-label">Amount</div>', unsafe_allow_html=True)
        amount = st.number_input(
            "amount", min_value=0.0, value=181_000.0, step=1000.0, label_visibility="collapsed"
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="field-label">Sender bal. before</div>', unsafe_allow_html=True)
            old_orig = st.number_input(
                "oldbalanceorig",
                min_value=0.0,
                value=181_000.0,
                step=1000.0,
                label_visibility="collapsed",
            )
        with col_b:
            st.markdown('<div class="field-label">Sender bal. after</div>', unsafe_allow_html=True)
            new_orig = st.number_input(
                "newbalanceorig",
                min_value=0.0,
                value=0.0,
                step=1000.0,
                label_visibility="collapsed",
            )

        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown(
                '<div class="field-label">Receiver bal. before</div>', unsafe_allow_html=True
            )
            old_dest = st.number_input(
                "oldbalancedest",
                min_value=0.0,
                value=0.0,
                step=1000.0,
                label_visibility="collapsed",
            )
        with col_d:
            st.markdown(
                '<div class="field-label">Receiver bal. after</div>', unsafe_allow_html=True
            )
            new_dest = st.number_input(
                "newbalancedest",
                min_value=0.0,
                value=0.0,
                step=1000.0,
                label_visibility="collapsed",
            )

        rule_flag = st.checkbox("Upstream rule engine already flagged this")
        submitted = st.form_submit_button("Assess risk", use_container_width=True)

    if submitted:
        st.session_state["score_payload"] = {
            "step": int(step),
            "types": ui.TRANSACTION_TYPES[type_label],
            "amount": float(amount),
            "oldbalanceorig": float(old_orig),
            "newbalanceorig": float(new_orig),
            "oldbalancedest": float(old_dest),
            "newbalancedest": float(new_dest),
            "isflaggedfraud": 1.0 if rule_flag else 0.0,
        }
        st.session_state["score_label"] = type_label
        st.session_state["score_reference"] = f"TXN-{uuid.uuid4().hex[:8].upper()}"

    payload = st.session_state.get("score_payload")
    if payload:
        delta = payload["newbalanceorig"] - payload["oldbalanceorig"]
        type_name = html.escape(st.session_state["score_label"])
        st.markdown(
            f"""
            <div class="summary-card">
              <div class="summary-title">Current transaction summary</div>
              <div class="summary-row"><span>Reference</span>
                <span class="summary-value">{st.session_state["score_reference"]}</span></div>
              <div class="summary-row"><span>Type</span>
                <span class="summary-value">{type_name}</span></div>
              <div class="summary-row"><span>Amount</span>
                <span class="summary-value">{ui.money(payload["amount"])}</span></div>
              <div class="summary-row"><span>Net liquidity change</span>
                <span class="summary-value {"negative" if delta < 0 else "positive"}">
                {ui.money(delta)}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with right:
    payload = st.session_state.get("score_payload")
    if not payload:
        st.markdown(
            '<div class="panel"><div class="panel-title">Awaiting input</div>'
            '<div class="panel-subtitle">Submit a transaction to score it.</div></div>',
            unsafe_allow_html=True,
        )
        st.stop()

    try:
        result = api.predict(payload)
    except api.ApiError as exc:
        st.error(str(exc))
        st.stop()

    probability = result["fraud_probability"]
    alerted = result["decision"] == "ALERT"
    badge = "ALERT" if alerted else "CLEARED"
    fill = (
        "linear-gradient(90deg, #5b080b, #d84c42)"
        if alerted
        else "linear-gradient(90deg, #103d2a, #35f29b)"
    )

    st.markdown(
        f"""
        <div class="panel {"critical" if alerted else ""}">
            <span class="risk-badge">{badge}</span>
            <div class="panel-title">Fraud probability</div>
            <div class="panel-subtitle">XGBoost, against the cost based cutoff</div>
            <div class="risk-number">{probability:.1f}<span>%</span></div>
            <div class="risk-meter">
                <div class="risk-fill" style="width: {min(probability, 100):.1f}%;
                    background: {fill};"></div>
            </div>
            <div class="summary-row" style="margin-top: 1.1rem;">
                <span>Alert threshold</span>
                <span class="summary-value">{result["threshold"]:.2f}%</span></div>
            <div class="summary-row"><span>Expected loss</span>
                <span class="summary-value">{ui.money(result["expected_loss"])}</span></div>
            <div class="summary-row"><span>Queue position ID</span>
                <span class="summary-value">#{result["alert_id"]}</span></div>
            <div class="summary-row"><span>Latency</span>
                <span class="summary-value">{result["latency_ms"]} ms</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    explanation = result.get("explanation", {})
    if explanation.get("available"):
        factors = explanation["factors"]
        largest = max(abs(f["impact"]) for f in factors) or 1
        rows = ""
        for factor in factors:
            impact = factor["impact"]
            width = max(6, min(100, abs(impact) / largest * 100))
            colour = "#c98f91" if impact > 0 else "#35f29b"
            sign = "+" if impact > 0 else ""
            rows += (
                '<div class="factor-row"><div class="factor-head">'
                f"<span>{html.escape(factor['feature'])}</span>"
                f'<span class="factor-impact" style="color: {colour};">'
                f"{sign}{impact:.3f}</span></div>"
                '<div class="bar-track">'
                f'<div class="bar-fill" style="width: {width}%; '
                f'background: {colour};"></div></div></div>'
            )
        st.markdown(
            '<div class="panel">'
            '<div class="panel-title">Feature attribution (SHAP)</div>'
            '<div class="panel-subtitle">Log-odds contribution for this '
            "transaction. Positive means more likely fraud.</div>"
            f"{rows}</div>",
            unsafe_allow_html=True,
        )
    else:
        # If SHAP didn't run, say so rather than showing something made up.
        st.markdown(
            f"""<div class="panel">
                <div class="panel-title">Feature attribution unavailable</div>
                <div class="panel-subtitle">
                  The score above is exact, but nothing could be worked out for
                  this one: {html.escape(str(explanation.get("reason")))}.
                </div></div>""",
            unsafe_allow_html=True,
        )
