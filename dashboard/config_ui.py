"""
dashboard/config_ui.py — Page Configuration and Global CSS.
Dark medical-grade theme with glassmorphic cards.
"""
import streamlit as st


def apply_page_config() -> None:
    """Sets Streamlit page config and injects global CSS."""
    st.set_page_config(
        page_title="CT QC Platform",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


_GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #0d1117; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #21262d;
}
[data-testid="stMetric"] {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 12px; padding: 16px !important;
}
[data-testid="stMetricLabel"] { font-size: 11px !important; color: #8b949e !important; }
[data-testid="stMetricValue"] { font-size: 20px !important; font-weight: 600 !important; }
.kpi-pass {
    background: linear-gradient(135deg, #0d2818 0%, #112a1c 100%);
    border: 1.5px solid #238636; border-radius: 14px;
    padding: 18px 16px 14px; text-align: center; transition: transform 0.2s;
}
.kpi-pass:hover { transform: translateY(-2px); }
.kpi-fail {
    background: linear-gradient(135deg, #2d0a0a 0%, #2d1111 100%);
    border: 1.5px solid #da3633; border-radius: 14px;
    padding: 18px 16px 14px; text-align: center; transition: transform 0.2s;
}
.kpi-fail:hover { transform: translateY(-2px); }
.kpi-warn {
    background: linear-gradient(135deg, #2d1d00 0%, #2d2000 100%);
    border: 1.5px solid #d29922; border-radius: 14px;
    padding: 18px 16px 14px; text-align: center;
}
.kpi-na {
    background: #161b22; border: 1.5px solid #21262d;
    border-radius: 14px; padding: 18px 16px 14px;
    text-align: center; opacity: 0.5;
}
.kpi-icon { font-size: 28px; margin-bottom: 6px; }
.kpi-label { font-size: 11px; font-weight: 500; letter-spacing: 0.06em;
             text-transform: uppercase; color: #8b949e; margin-bottom: 4px; }
.kpi-value { font-size: 18px; font-weight: 700; margin-bottom: 2px; }
.kpi-sub   { font-size: 10px; color: #8b949e; }
.section-hdr {
    font-size: 10px; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #8b949e;
    border-bottom: 1px solid #21262d;
    padding-bottom: 6px; margin-bottom: 14px; margin-top: 8px;
}
.hw-critical { background:#2d0a0a; border:1.5px solid #da3633;
               border-radius:10px; padding:14px; margin:6px 0; }
.hw-warning  { background:#2d1a00; border:1.5px solid #d29922;
               border-radius:10px; padding:14px; margin:6px 0; }
.hw-monitor  { background:#1c1a00; border:1.5px solid #c9a227;
               border-radius:10px; padding:14px; margin:6px 0; }
.hw-stable   { background:#0d2818; border:1.5px solid #238636;
               border-radius:10px; padding:14px; margin:6px 0; }
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: #161b22; border-radius: 10px; padding: 4px;
}
.stTabs [data-baseweb="tab"] { border-radius:7px; color:#8b949e; font-size:13px; }
.stTabs [aria-selected="true"] {
    background:#21262d !important; color:#f0f6fc !important;
}
[data-testid="stFileUploader"] {
    border: 2px dashed #21262d !important;
    border-radius: 12px !important; background: #161b22 !important;
}
[data-testid="stDataFrame"] {
    border: 1px solid #21262d !important; border-radius: 8px !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1f6feb 0%, #1158c7 100%);
    border: none; border-radius: 8px;
    font-weight: 600; font-size: 13px; padding: 8px 24px;
}
</style>
"""
