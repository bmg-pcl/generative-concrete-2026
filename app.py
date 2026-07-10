import streamlit as st

from ui import sidebar, state
from ui.config import render_config
from ui.compare import render_compare
from ui.inverse import render_inverse
from ui.pareto import render_pareto
from ui.calibration import render_calibration
from ui.workflow import render_workflow
from ui.reference import render_technical_report, render_references

st.set_page_config(page_title="Generative Mix Design", layout="wide", initial_sidebar_state="expanded")

# --- Custom classes only ---
# Base dark theme lives in .streamlit/config.toml (supported theming API). Here we
# keep just the custom classes used via markdown, which don't target Streamlit
# internals and so won't break on library upgrades.
st.markdown("""
<style>
    .main-title { font-size: 3rem; font-weight: 800; background: -webkit-linear-gradient(#00E676, #2979FF); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .footnote { font-size: 0.75rem; color: #888; line-height: 1.4; margin-top: 20px; }
</style>
""", unsafe_allow_html=True)

# --- Initialize models + presets (cached) ---
status_container = st.empty()
status_container.info("Initializing models...")
predictor, bayesian = state.load_resources()
presets = state.get_preset_mixtures()
status_container.empty()

state.init_session_state()

# --- App Header ---
st.markdown('<h1 class="main-title">Generative Mix Design</h1>', unsafe_allow_html=True)
st.markdown("AI-powered concrete formulation: prediction, optimization, and inverse design.")

sidebar.render()

# --- Main Layout ---
tab1, tab2, tab3, tab4, tab_config, tab_workflow, tab5, tab6 = st.tabs([
    "Compare Mixes", "Inverse Design", "Pareto Optimization",
    "Calibration", "Config", "Workflow", "Technical Report", "References",
])

# render_config builds the AppContext the other tabs consume; passing ctx into the
# other renderers is what enforces "Config is read first" — it's a dataflow
# dependency now, not a comment-guarded ordering.
with tab_config:
    ctx = render_config(predictor, bayesian, presets)
with tab1:
    render_compare(ctx)
with tab2:
    render_inverse(ctx)
with tab3:
    render_pareto(ctx)
with tab4:
    render_calibration(ctx)
with tab_workflow:
    render_workflow()
with tab5:
    render_technical_report()
with tab6:
    render_references()
