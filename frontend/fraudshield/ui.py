"""Bits shared between the pages."""

from __future__ import annotations

import html

import streamlit as st

from . import api

TRANSACTION_TYPES = {
    "Cash In": 0,
    "Cash Out": 1,
    "Debit": 2,
    "Payment": 3,
    "Transfer": 4,
}
TYPE_LABELS = {code: name for name, code in TRANSACTION_TYPES.items()}


def money(value: float) -> str:
    """Short currency format for the tiles."""
    value = float(value)
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= limit:
            return f"${value / limit:,.1f}{suffix}"
    return f"${value:,.0f}"


def topbar(subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="topbar">
            <div class="brand">
                <div class="brand-mark">FS</div>
                <div>
                    <div class="page-title">FraudShield Intelligence</div>
                    <div class="panel-subtitle">{html.escape(subtitle)}</div>
                </div>
            </div>
            <div class="system-live"><span class="live-dot"></span>System Live</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_cards(cards: list[dict]) -> None:
    """cards: [{label, value, hint?, tone?}], tone is 'good', 'bad' or None."""
    tiles = "".join(
        # One line per tile. st.markdown only strips the shared indent, so any
        # leftover leading space in generated HTML turns into a code block and
        # shows up as raw markup on the page.
        '<div class="metric-card">'
        f'<div class="metric-label">{html.escape(card["label"])}</div>'
        f'<div class="metric-value {card.get("tone") or ""}">'
        f"{html.escape(str(card['value']))}</div>"
        f'<div class="metric-hint">{html.escape(card.get("hint", ""))}</div>'
        "</div>"
        for card in cards
    )
    st.markdown(f'<div class="metric-grid">{tiles}</div>', unsafe_allow_html=True)


def note(markup: str) -> None:
    """Explanation box. Text comes from the pages, not from user input."""
    st.markdown(f'<div class="note">{markup}</div>', unsafe_allow_html=True)


def pill(text: str, band: str) -> str:
    return f'<span class="pill {html.escape(band)}">{html.escape(text)}</span>'


def connection_banner() -> dict | None:
    """Checks the backend is up. Returns the health response, or None if it isn't."""
    try:
        status = api.health()
    except api.ApiError as exc:
        st.error(str(exc))
        st.caption(f"Configured endpoint: {api.base_url()} (set the API_URL env var)")
        return None

    if not status.get("explanations_available"):
        st.warning(
            "SHAP explanations are unavailable on this deployment "
            f"({status.get('explanations_unavailable_reason')}). Scores are still "
            "exact; individual predictions will show no factor breakdown."
        )
    return status
