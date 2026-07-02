"""Sidebar: step-by-step guidance + session save/restore."""
import json

import streamlit as st

from src.ui_logic import validate_session_state
from ui.state import get_state_json, load_mix_into


def render():
    with st.sidebar:
        st.header("How to use")
        st.caption("Collapse this panel with the arrow at its top-right; reopen it from the same edge.")
        with st.expander("Step-by-step guide", expanded=True):
            st.markdown(
                "**1. Config** — set material costs and choose the carbon model "
                "(Simple or Advanced). These apply everywhere.\n\n"
                "**2. Compare Mixes** — tune Mix A and Mix B and read predicted strength, "
                "carbon, and cost side by side.\n\n"
                "**3. Inverse Design** — enter a target strength, pick a backend, and get a "
                "recommended recipe you can load into Mix A or B.\n\n"
                "**4. Pareto Optimization** — GA/SA (weighted) or NSGA-II/III (true trade-off "
                "front) across strength / carbon / cost.\n\n"
                "**5. Calibration** — upload your own lab results (CSV) to retrain the model "
                "on your materials.\n\n"
                "See the **Workflow** tab for the full walkthrough and when to use which tool."
            )
        st.divider()
        with st.expander("Session save / restore", expanded=False):
            st.caption("Save or restore your complete setup as a JSON file.")
            st.download_button(
                label="Export Session (JSON)", data=get_state_json(),
                file_name="gmd_session.json", mime="application/json",
            )
            uploaded_state = st.file_uploader("Import Session", type="json")
            if uploaded_state:
                try:
                    data = json.load(uploaded_state)
                except json.JSONDecodeError:
                    data = None
                error = validate_session_state(data) if data is not None else "File is not valid JSON."
                if error:
                    st.error(f"Import failed: {error}")
                else:
                    # Sidebar runs before the Compare tab instantiates the sliders, so
                    # writing the keyed values here is safe (precedes widget instantiation).
                    load_mix_into("A", data["mix_a"])
                    load_mix_into("B", data["mix_b"])
                    st.session_state.costs = data["costs"]
                    if "carbon_factors" in data:   # v2; v1 files simply keep the defaults
                        st.session_state.carbon_factors = data["carbon_factors"]
                    if "exotic_a" in data:
                        st.session_state.exotic_a = data["exotic_a"]
                        st.session_state.exotic_b = data["exotic_b"]
                    st.success("Session Imported!")
