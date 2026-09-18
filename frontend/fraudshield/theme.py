"""Styling for the dashboard.

Same CSS as the original single page version, so the pages still match.
"""

import streamlit as st

CSS = """
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
"""


def apply() -> None:
    """Adds the stylesheet. Called once per page."""
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(EXTRA_CSS, unsafe_allow_html=True)


# Extra styles for the queue, batch, drift and model card pages.
EXTRA_CSS = """
    <style>
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 0.85rem;
        margin: 0.6rem 0 1.2rem;
    }
    .metric-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 7px;
        padding: 0.95rem 1.05rem;
    }
    .metric-label {
        color: var(--muted);
        font-size: 0.68rem;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        font-weight: 600;
    }
    .metric-value {
        color: var(--text);
        font-size: 1.55rem;
        font-weight: 700;
        line-height: 1.25;
        margin-top: 0.35rem;
    }
    .metric-hint { color: var(--muted); font-size: 0.72rem; margin-top: 0.2rem; }
    .metric-value.good { color: var(--green); }
    .metric-value.bad { color: var(--red-strong); }

    .alert-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-left: 3px solid var(--red-strong);
        border-radius: 6px;
        padding: 0.9rem 1.05rem;
        margin-bottom: 0.55rem;
    }
    .alert-head {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 1rem;
        flex-wrap: wrap;
    }
    .alert-loss { color: var(--red); font-weight: 700; font-size: 1.05rem; }
    .alert-meta { color: var(--muted); font-size: 0.74rem; }

    .pill {
        display: inline-block;
        padding: 0.16rem 0.55rem;
        border-radius: 999px;
        font-size: 0.66rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        border: 1px solid var(--line);
    }
    .pill.stable { color: var(--green); border-color: rgba(53, 242, 155, 0.4); }
    .pill.moderate { color: #f5c76a; border-color: rgba(245, 199, 106, 0.4); }
    .pill.significant { color: var(--red); border-color: rgba(214, 72, 62, 0.5); }

    .note {
        background: var(--panel-2);
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: 0.8rem 1rem;
        color: var(--muted);
        font-size: 0.82rem;
        line-height: 1.55;
        margin-bottom: 0.9rem;
    }
    .note strong { color: var(--text); }
    </style>
"""
