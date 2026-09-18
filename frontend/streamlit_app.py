"""Dashboard entry point.

Five pages: score a transaction, work the queue, score a file, check drift,
read the model card.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="FraudShield Intelligence",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded",
)

from fraudshield import theme  # noqa: E402  (must follow set_page_config)

theme.apply()

pages = [
    st.Page("pages/1_score.py", title="Risk console", icon=":material/search:", default=True),
    st.Page("pages/2_queue.py", title="Review queue", icon=":material/inbox:"),
    st.Page("pages/3_batch.py", title="Batch scoring", icon=":material/upload_file:"),
    st.Page("pages/4_drift.py", title="Drift monitor", icon=":material/monitoring:"),
    st.Page("pages/5_model_card.py", title="Model card", icon=":material/description:"),
]

st.navigation(pages).run()
