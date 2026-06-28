import os
from datetime import datetime

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

TRANSACTION_TYPES = {
    "Cash In": 0,
    "Cash Out": 1,
    "Debit": 2,
    "Payment": 3,
    "Transfer": 4,
}


st.set_page_config(
    page_title="FraudShield Intelligence",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #05070b;
        --panel: #0d1320;
        --panel-2: #111827;
        --panel-3: #1f2430;
        --text: #dbe6ff;
        --muted: #93a4bd;
        --line: #243044;
        --blue: #3b82f6;
        --blue-2: #60a5fa;
        --green: #35f29b;
        --red: #ff9b95;
        --red-strong: #d6483e;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    .block-container {
        max-width: 1280px;
        padding: 1.1rem 1.7rem 2.2rem;
    }

    header[data-testid="stHeader"], [data-testid="stToolbar"], footer {
        display: none;
    }

    div[data-testid="stVerticalBlock"] {
        gap: 1.05rem;
    }

    div[data-testid="stForm"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 7px;
        padding: 1.45rem;
        box-shadow: 0 20px 45px rgba(0, 0, 0, 0.22);
    }

    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 0 0.9rem;
        border-bottom: 1px solid #151c2a;
        margin-bottom: 0.35rem;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        font-weight: 800;
        color: #eff6ff;
        font-size: 1.12rem;
    }

    .brand-mark {
        width: 32px;
        height: 32px;
        border-radius: 5px;
        display: grid;
        place-items: center;
        background: linear-gradient(135deg, #2f7eff, #69a7ff);
        color: #06101f;
        font-weight: 900;
        box-shadow: 0 0 22px rgba(59, 130, 246, 0.32);
    }

    .brand span {
        color: var(--blue-2);
    }

    .system-live {
        border: 1px solid #273142;
        background: #0a0d14;
        color: #aebbd0;
        padding: 0.45rem 1rem;
        border-radius: 999px;
        font-size: 0.72rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }

    .live-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 999px;
        background: var(--green);
        margin-right: 0.6rem;
        box-shadow: 0 0 14px rgba(53, 242, 155, 0.85);
    }

    .page-title {
        margin: 1.2rem 0 1.6rem;
        display: flex;
        align-items: end;
        justify-content: flex-start;
        gap: 1rem;
    }

    .page-title h1 {
        margin: 0;
        color: #dbe6ff;
        font-size: clamp(2rem, 5vw, 3rem);
        line-height: 1;
        font-weight: 800;
        letter-spacing: 0;
    }

    .page-title h1 span {
        color: var(--blue);
    }

    .page-title p {
        color: var(--muted);
        margin: 0.45rem 0 0;
        font-size: 1.03rem;
    }

    .panel {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 7px;
        padding: 1.45rem;
        box-shadow: 0 20px 45px rgba(0, 0, 0, 0.22);
    }

    .panel.critical {
        border-color: rgba(255, 155, 149, 0.7);
    }

    .panel-title {
        color: #dfe8ff;
        font-size: 1.28rem;
        font-weight: 800;
        margin-bottom: 0.85rem;
    }

    .panel-subtitle {
        color: var(--muted);
        font-size: 0.92rem;
        font-weight: 700;
        margin-bottom: 1.35rem;
    }

    .field-label {
        color: #9ba9bd;
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.38rem;
    }

    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input {
        background: #070a10 !important;
        border: 1px solid #1e293b !important;
        border-radius: 4px !important;
        color: #e5edff !important;
        min-height: 42px;
    }

    div[data-testid="stNumberInput"] button {
        background: #0b1018 !important;
        border-color: #1e293b !important;
        color: #dbe6ff !important;
    }

    .stButton > button {
        width: 100%;
        background: #1d355d;
        color: #5da2ff;
        border: 1px solid #315d9c;
        border-radius: 7px;
        min-height: 56px;
        font-size: 0.98rem;
        font-weight: 800;
    }

    .stButton > button:hover {
        background: #244577;
        border-color: var(--blue);
        color: #b9d6ff;
    }

    .summary-card {
        margin-top: 1.4rem;
        background: #0d1320;
        border: 1px solid var(--line);
        border-radius: 7px;
        padding: 1.25rem;
    }

    .summary-title {
        font-size: 0.78rem;
        color: #dbe6ff;
        font-weight: 800;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    .summary-row {
        display: flex;
        justify-content: space-between;
        border-bottom: 1px solid #1a2333;
        padding: 0.48rem 0;
        color: var(--muted);
        font-size: 0.86rem;
    }

    .summary-row:last-child {
        border-bottom: 0;
    }

    .summary-value {
        color: #dbe6ff;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }

    .positive {
        color: var(--green);
    }

    .negative {
        color: var(--red);
    }

    .risk-badge {
        display: inline-flex;
        float: right;
        border-radius: 999px;
        border: 1px solid rgba(255, 155, 149, 0.35);
        background: rgba(255, 155, 149, 0.12);
        color: #ffc5c0;
        padding: 0.36rem 0.95rem;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.04em;
    }

    .risk-number {
        text-align: center;
        font-size: clamp(2.1rem, 7vw, 3.8rem);
        font-weight: 800;
        line-height: 1;
        color: #ffd5d1;
        margin: 1.9rem 0 0.7rem;
    }

    .risk-number span {
        font-size: 1.15rem;
        margin-left: 0.2rem;
    }

    .risk-meter {
        height: 38px;
        background: #070a10;
        border: 1px solid #1f2937;
        border-radius: 8px;
        overflow: hidden;
        margin: 1.15rem 0 0.8rem;
    }

    .risk-fill {
        height: 100%;
        background: linear-gradient(90deg, #5b080b, #d84c42);
        display: flex;
        align-items: center;
        justify-content: flex-end;
    }

    .factor-row {
        margin-bottom: 0.95rem;
    }

    .factor-head {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        color: #dbe6ff;
        font-size: 0.92rem;
        margin-bottom: 0.32rem;
    }

    .factor-impact {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-weight: 800;
    }

    .bar-track {
        height: 5px;
        background: #070a10;
        border-radius: 999px;
        overflow: hidden;
    }

    .bar-fill {
        height: 100%;
        border-radius: 999px;
    }

    .status {
        display: inline-block;
        border-radius: 3px;
        padding: 0.25rem 0.55rem;
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.04em;
    }

    .status.blocked {
        color: #ffc5c0;
        border: 1px solid rgba(255, 155, 149, 0.35);
        background: rgba(255, 155, 149, 0.12);
    }

    .status.cleared {
        color: #35f29b;
        border: 1px solid rgba(53, 242, 155, 0.28);
        background: rgba(53, 242, 155, 0.08);
    }

    .status.flagged {
        color: #7bb5ff;
        border: 1px solid rgba(59, 130, 246, 0.35);
        background: rgba(59, 130, 246, 0.11);
    }

    @media (max-width: 900px) {
        .page-title {
            align-items: flex-start;
            flex-direction: column;
        }

    }
    </style>
    """,
    unsafe_allow_html=True,
)


def money(value):
    return f"${value:,.2f}"


def get_risk_score(result):
    return max(0, min(100, float(result.get("confidence", 0))))


def get_status(result):
    if result.get("status"):
        return result["status"]

    prediction = result.get("prediction", "")
    if "Fraudulent" in prediction and "Not Fraudulent" not in prediction:
        return "FRAUDULENT"
    if prediction:
        return "CLEARED"
    return "PENDING"


def render_topbar():
    st.markdown(
        """
        <div class="topbar">
            <div class="brand">
                <div class="brand-mark">F</div>
                <div>FraudShield <span>Intelligence</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_title():
    st.markdown(
        """
        <div class="page-title">
            <div>
                <h1>FraudShield Risk <span>Console</span></h1>
                <p>Real-time risk assessment and feature attribution</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary(reference_id, selected_type, amount, sender_delta, velocity):
    delta_class = "negative" if sender_delta < 0 else "positive"
    st.markdown(
        f"""
        <div class="summary-card">
            <div class="summary-title">Current Transaction Summary</div>
            <div class="summary-row"><span>Reference ID</span><span class="summary-value">{reference_id}</span></div>
            <div class="summary-row"><span>Transaction Type</span><span class="summary-value">{selected_type}</span></div>
            <div class="summary-row"><span>Amount</span><span class="summary-value">{money(amount)}</span></div>
            <div class="summary-row"><span>Net Liquidity Change</span><span class="summary-value {delta_class}">{money(sender_delta)}</span></div>
            <div class="summary-row"><span>Velocity Check</span><span class="summary-value positive">{velocity}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_panel(result):
    confidence = get_risk_score(result)
    display_score = int(round(confidence))
    status = get_status(result)
    if status == "PENDING":
        badge = "PENDING"
    else:
        badge = "FRAUDULENT" if status in ["CRITICAL RISK", "FRAUDULENT"] else "CLEARED"
    border_class = "critical" if badge == "FRAUDULENT" else ""
    if badge == "PENDING":
        fill_color = "linear-gradient(90deg, #1d4ed8, #60a5fa)"
    elif badge == "CLEARED":
        fill_color = "linear-gradient(90deg, #103d2a, #35f29b)"
    else:
        fill_color = "linear-gradient(90deg, #5b080b, #d84c42)"

    st.markdown(
        f"""
        <div class="panel {border_class}">
            <span class="risk-badge">{badge}</span>
            <div class="panel-title">Risk Score Prediction</div>
            <div class="panel-subtitle">XGBoost confidence</div>
            <div class="risk-number">{display_score}<span>%</span></div>
            <div class="risk-meter">
                <div class="risk-fill" style="width: {display_score}%; background: {fill_color};"></div>
            </div>
            <div class="summary-row" style="margin-top: 1.1rem;"><span>Model</span><span class="summary-value">{result.get("model", "XGBoost")}</span></div>
            <div class="summary-row"><span>XGBoost Confidence</span><span class="summary-value">{confidence:.2f}%</span></div>
            <div class="summary-row"><span>Prediction Latency</span><span class="summary-value">{result.get("prediction_time_ms", 0)} ms</span></div>
            <div class="summary-row"><span>Decision</span><span class="summary-value">{badge}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_attribution(result):
    factors = result.get("top_factors", [])
    if not factors:
        factors = [
            {"feature": "Transaction Amount", "impact": 0.0},
            {"feature": "Sender Balance Movement", "impact": 0.0},
            {"feature": "Transaction Type", "impact": 0.0},
        ]

    max_impact = max(abs(item["impact"]) for item in factors) or 1
    factor_html = ""

    for item in factors:
        impact = float(item["impact"])
        width = max(6, min(100, abs(impact) / max_impact * 100))
        color = "#35f29b" if impact < 0 else "#c98f91"
        sign = "+" if impact > 0 else ""
        factor_html += f"""
        <div class="factor-row">
            <div class="factor-head">
                <span>{item["feature"]}</span>
                <span class="factor-impact" style="color: {color};">{sign}{impact:.3f}</span>
            </div>
            <div class="bar-track"><div class="bar-fill" style="width: {width}%; background: {color};"></div></div>
        </div>
        """

    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-title">Feature Attribution (SHAP)</div>
            <div class="panel-subtitle">Top model drivers for this transaction</div>
            {factor_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


render_topbar()
render_title()

left, right = st.columns([0.42, 0.58], gap="large")

with left:
    with st.form("risk_form"):
        st.markdown('<div class="panel-title">Input Parameters</div>', unsafe_allow_html=True)
        st.markdown('<div class="field-label">Transaction Hour</div>', unsafe_allow_html=True)
        step = st.number_input(
            "Transaction Hour",
            min_value=0,
            max_value=24,
            value=0,
            step=1,
            label_visibility="collapsed",
        )

        st.markdown('<div class="field-label">Transaction Type</div>', unsafe_allow_html=True)
        selected_type = st.selectbox(
            "Transaction Type",
            list(TRANSACTION_TYPES.keys()),
            index=0,
            label_visibility="collapsed",
        )

        st.markdown('<div class="field-label">Transaction Amount</div>', unsafe_allow_html=True)
        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=0.0,
            step=100.0,
            label_visibility="collapsed",
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="field-label">Sender Bal. Before</div>', unsafe_allow_html=True)
            oldbalanceorig = st.number_input(
                "Sender Balance Before",
                min_value=0.0,
                value=0.0,
                step=100.0,
                label_visibility="collapsed",
            )
        with c2:
            st.markdown('<div class="field-label">Sender Bal. After</div>', unsafe_allow_html=True)
            newbalanceorig = st.number_input(
                "Sender Balance After",
                min_value=0.0,
                value=0.0,
                step=100.0,
                label_visibility="collapsed",
            )

        c3, c4 = st.columns(2)
        with c3:
            st.markdown('<div class="field-label">Receiver Bal. Before</div>', unsafe_allow_html=True)
            oldbalancedest = st.number_input(
                "Receiver Balance Before",
                min_value=0.0,
                value=0.0,
                step=100.0,
                label_visibility="collapsed",
            )
        with c4:
            st.markdown('<div class="field-label">Receiver Bal. After</div>', unsafe_allow_html=True)
            newbalancedest = st.number_input(
                "Receiver Balance After",
                min_value=0.0,
                value=0.0,
                step=100.0,
                label_visibility="collapsed",
            )

        submitted = st.form_submit_button("Execute Risk Scan")

    summary_slot = st.empty()

types = TRANSACTION_TYPES[selected_type]
isflaggedfraud = 1 if amount >= 200000 else 0

payload = {
    "step": step,
    "types": types,
    "amount": amount,
    "oldbalanceorig": oldbalanceorig,
    "newbalanceorig": newbalanceorig,
    "oldbalancedest": oldbalancedest,
    "newbalancedest": newbalancedest,
    "isflaggedfraud": isflaggedfraud,
}

zero_result = {
    "prediction": "Not Fraudulent Transaction",
    "status": "PENDING",
    "confidence": 0.0,
    "fraud_probability": 0.0,
    "model": "XGBoost",
    "prediction_time_ms": 0,
    "top_factors": [
        {"feature": "Transaction Amount", "impact": 0.0},
        {"feature": "Sender Balance Movement", "impact": 0.0},
        {"feature": "Transaction Type", "impact": 0.0},
    ],
}

result = st.session_state.get("last_result", zero_result)

if "last_transaction" not in st.session_state:
    st.session_state["last_transaction"] = {
        "reference_id": "#TX-PENDING",
        "selected_type": "Cash In",
        "amount": 0.0,
        "sender_delta": 0.0,
        "velocity": "Normal",
    }

if submitted:
    try:
        response = requests.post( "https://fraudshield-intelligence.onrender.com/", json=payload)
        if response.status_code != 200:
            st.error(f"FastAPI Error ({response.status_code})")
            st.code(response.text)
            st.stop()
        result = response.json()
        st.session_state["last_result"] = result
        tx_id = f"#TX-{datetime.now().strftime('%H%M%S')}"
        sender_delta = newbalanceorig - oldbalanceorig
        st.session_state["last_transaction"] = {
            "reference_id": tx_id,
            "selected_type": selected_type,
            "amount": amount,
            "sender_delta": sender_delta,
            "velocity": "Normal",
        }
    except Exception as exc:
        st.error("Could not connect to the FastAPI server.")
        st.caption(f"Checked endpoint: {API_URL}")
        st.exception(exc)

summary = st.session_state["last_transaction"]
with summary_slot.container():
    render_summary(
        summary["reference_id"],
        summary["selected_type"],
        summary["amount"],
        summary["sender_delta"],
        summary["velocity"],
    )

with right:
    render_risk_panel(result)
    render_attribution(result)




