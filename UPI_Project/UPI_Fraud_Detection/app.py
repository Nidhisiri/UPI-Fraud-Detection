"""UPI Fraud Detection Platform - Streamlit entry point.
Run:  streamlit run app.py     (from inside the UPI_Fraud_Detection folder)"""
import streamlit as st

st.set_page_config(page_title="UPI Fraud Detection Platform", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")

from upi_app import backend as B                       # noqa: E402
from upi_app.theme import html, inject_css, sidebar_brand   # noqa: E402

inject_css()

missing = B.missing_files()
if missing:
    st.error("Some files exported by the notebook were not found:\n\n" + "\n".join(f"- `{m}`" for m in missing))
    st.info("Run section 14 of your notebook (it creates `models/` and `data/`), and place this app's files in the same "
            "`UPI_Fraud_Detection` folder. See HOW_TO_RUN.md.")
    st.stop()

from upi_app import pages_analytics as PA, pages_common as PC, pages_user as PU   # noqa: E402

USER, TECH = "Everyday user", "Analytics"
PAGES = {
    "Overview": PC.page_overview,
    "Check Your UPI Transaction": PU.page_checker,
    "Transaction History": PU.page_history,
    "How the AI Decides": PU.page_how_ai,
    "Data Exploration": PA.page_eda,
    "Preprocessing Pipeline": PA.page_preprocessing,
    "Model Performance": PA.page_model,
    "Explainable AI (SHAP)": PA.page_shap,
    "About the Project": PC.page_about,
}
MODE_PAGES = {
    USER: ["Overview", "Check Your UPI Transaction", "Transaction History", "How the AI Decides", "About the Project"],
    TECH: ["Overview", "Data Exploration", "Preprocessing Pipeline", "Model Performance", "Explainable AI (SHAP)", "About the Project"],
}

st.session_state.setdefault("mode", USER)
with st.sidebar:
    sidebar_brand()
    st.markdown('<div class="side-label">Choose your experience</div>', unsafe_allow_html=True)
    mode = st.radio("Mode", [USER, TECH], key="mode", label_visibility="collapsed", horizontal=True)
    options = MODE_PAGES[mode]
    if st.session_state.get("nav_page") not in options:
        st.session_state["nav_page"] = options[0] if options[0] != "Overview" else options[1]
    st.markdown('<div class="side-label">Navigate</div>', unsafe_allow_html=True)
    page = st.radio("Page", options, key="nav_page", label_visibility="collapsed")
    st.markdown('<div class="side-label">Everyday user = simple checker · Analytics = full technical view</div>', unsafe_allow_html=True)
    if mode == USER and st.session_state.get("history"):
        st.caption(f"{len(st.session_state['history'])} transaction(s) analyzed this session")

try:
    PAGES[page]()
except Exception as e:                                   # keep the app usable if one page fails
    st.error(f"This page could not be displayed: {e}")
    st.exception(e)
