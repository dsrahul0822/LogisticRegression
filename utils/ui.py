"""
ui.py
-----
Small UI helpers: custom CSS, section headers, and metric cards.
The colour theme follows the client brief:
    primary  -> #00E676 (green)
    accent   -> #FF2D95 (pink)
"""

import streamlit as st

PRIMARY = "#00E676"
ACCENT = "#FF2D95"
BG_DARK = "#0E1117"
BG_CARD = "#161A23"
TEXT = "#E6E6E6"
SUBTEXT = "#9BA3AF"


CUSTOM_CSS = f"""
<style>
    /* ---------- global ---------- */
    .stApp {{
        background: linear-gradient(180deg, {BG_DARK} 0%, #111522 100%);
        color: {TEXT};
    }}
    section[data-testid="stSidebar"] {{
        background: #0A0D14;
        border-right: 1px solid #1F2430;
    }}
    section[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}

    /* ---------- title / header ---------- */
    .app-title {{
        font-size: 34px;
        font-weight: 800;
        background: linear-gradient(90deg, {PRIMARY}, {ACCENT});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }}
    .app-subtitle {{
        color: {SUBTEXT};
        font-size: 14px;
        margin-top: 2px;
        margin-bottom: 18px;
    }}
    .page-header {{
        font-size: 26px;
        font-weight: 700;
        color: {PRIMARY};
        margin: 0 0 4px 0;
    }}
    .page-sub {{
        color: {SUBTEXT};
        font-size: 13px;
        margin-bottom: 18px;
    }}

    /* ---------- cards ---------- */
    .card {{
        background: {BG_CARD};
        border: 1px solid #1F2430;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.35);
        margin-bottom: 14px;
    }}
    .metric-card {{
        background: {BG_CARD};
        border: 1px solid #1F2430;
        border-left: 4px solid {PRIMARY};
        border-radius: 12px;
        padding: 14px 18px;
    }}
    .metric-label {{
        color: {SUBTEXT};
        font-size: 12px;
        letter-spacing: .6px;
        text-transform: uppercase;
    }}
    .metric-value {{
        color: {TEXT};
        font-size: 26px;
        font-weight: 700;
        margin-top: 2px;
    }}

    /* ---------- prediction boxes ---------- */
    .pred-box {{
        border-radius: 16px;
        padding: 22px 26px;
        text-align: center;
        font-weight: 700;
        font-size: 22px;
        margin: 10px 0;
    }}
    .pred-approved {{
        background: rgba(0, 230, 118, 0.10);
        color: {PRIMARY};
        border: 1px solid {PRIMARY};
    }}
    .pred-rejected {{
        background: rgba(255, 45, 149, 0.10);
        color: {ACCENT};
        border: 1px solid {ACCENT};
    }}

    /* ---------- streamlit tweaks ---------- */
    .stButton > button {{
        background: linear-gradient(90deg, {PRIMARY}, #00b85d);
        color: #0A0D14;
        border: 0;
        border-radius: 10px;
        padding: 8px 18px;
        font-weight: 700;
    }}
    .stButton > button:hover {{ filter: brightness(1.08); }}
    .stDownloadButton > button {{
        background: linear-gradient(90deg, {ACCENT}, #c9176f);
        color: white;
        border: 0;
        border-radius: 10px;
        font-weight: 700;
    }}
    .stDataFrame {{ border-radius: 10px; overflow: hidden; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def app_header() -> None:
    st.markdown(
        "<div class='app-title'>🏦 Banking Loan Prediction System</div>"
        "<div class='app-subtitle'>Logistic Regression · End-to-end ML dashboard</div>",
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"<div class='page-header'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='page-sub'>{subtitle}</div>", unsafe_allow_html=True)


def metric_card(col, label: str, value: str) -> None:
    col.markdown(
        f"<div class='metric-card'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
