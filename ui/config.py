"""Config tab — costing, carbon model, and experimental options.

`render_config()` renders the tab and returns the AppContext the other tabs consume;
building the context here is what enforces "Config is read before the other tabs".
"""
from datetime import datetime, timezone

import streamlit as st

from src.exotics import EXOTIC_STRENGTH_DISCLAIMER
from ui.context import AppContext


def render_config(predictor, bayesian, presets) -> AppContext:
    st.header("Configuration")
    st.caption("Costing, carbon model, and experimental options — applied across all tabs.")
    cfg_costs, cfg_model = st.columns(2)
    with cfg_costs:
        st.subheader("Material Costs")
        st.caption("Costs in $ per kilogram.")
        for mat in st.session_state.costs:
            st.session_state.costs[mat] = st.number_input(
                f"{mat.replace('_', ' ').title()} ($/kg)",
                value=st.session_state.costs[mat], format="%.4f",
            )
        with st.expander("Emission factors (kg CO₂ / kg material)"):
            st.caption("Regional/producer-specific — defaults from the ICE database & WBCSD/CSI protocol.")
            for mat in st.session_state.carbon_factors:
                # Keyed-only (no value=): the cf_ key IS the live value, initialised in
                # init_session_state and written by session import — same single-source
                # pattern as the mix sliders (see R4.1 / R5.1).
                st.session_state.carbon_factors[mat] = st.number_input(
                    f"{mat.replace('_', ' ').title()} (kg CO₂/kg)",
                    format="%.4f", key=f"cf_{mat}",
                )
    with cfg_model:
        st.subheader("Carbon & Analysis Model")
        chemistry_mode = st.radio(
            "Carbon model",
            ["Simple (Linear)", "Advanced (Molecular)"],
            help="Simple uses mass x factor. Advanced uses Bogue calculations and clinker chemistry.",
        )
        use_advanced_chemistry = (chemistry_mode == "Advanced (Molecular)")

        transport_km = st.number_input(
            "Transport distance (km)", 0, 2000, 0,
            help="Round-trip haul from plant to site. Adds ~0.1 kg CO₂ per tonne per km.",
        )
        cement_source = st.selectbox(
            "Clinker / cement source", ["OPC (Portland)", "LC3 (limestone calcined clay)"],
            help="Advanced tier only: sets the clinker factor (OPC ≈ 0.95, LC3 ≈ 0.50).",
        )
        cement_type = "LC3" if cement_source.startswith("LC3") else "OPC"
        # One carbon config threaded to every carbon computation across the tabs.
        carbon_kwargs = {
            "transport_km": float(transport_km),
            "cement_type": cement_type,
            "factors": st.session_state.carbon_factors,
        }

        st.divider()
        st.subheader("Optimization")
        robust_mode = st.toggle(
            "Robust mode (optimize the guaranteed-strength bound, stay in-data)",
            value=True,
            help="ON: inverse design and Pareto optimize the conformal LOWER bound of "
                 "strength and keep candidates inside the trusted data region "
                 "(NSGA enforces it as a constraint). OFF: optimize the mean prediction, "
                 "which can chase confident-but-unsupported optima.",
        )
        fix_age = st.checkbox(
            "Fix design age (inverse design & Pareto)", value=True,
            help="ON: hold age fixed so the optimizer can't hit a strength target by "
                 "prescribing a long cure (age is a condition, not a design variable). "
                 "The trained flow can't honor a fixed age, so pinning routes to the "
                 "GA/ACO designers. OFF: age is free.",
        )
        design_age = st.number_input("Design age (days)", 1, 365, 28) if fix_age else None

        st.divider()
        st.subheader("Exotic Strength Model")
        exotic_strength_enabled = st.toggle(
            "Include exotics in predicted strength (experimental)",
            value=False,
            help=(
                "OFF (default): exotic admixtures affect only cost and carbon. The strength "
                "model is trained on the UCI dataset, which contains none of these materials, "
                "so it cannot predict their effect. ON: a placeholder linear estimate adds "
                "their contribution to strength - UNVALIDATED, to be replaced when real exotic "
                "strength data is available."
            ),
        )
        if exotic_strength_enabled:
            st.warning(EXOTIC_STRENGTH_DISCLAIMER)

    ticket_config = {
        **carbon_kwargs, "advanced": use_advanced_chemistry, "costs": st.session_state.costs,
        "robust": robust_mode,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return AppContext(
        predictor=predictor, bayesian=bayesian, presets=presets,
        costs=st.session_state.costs, carbon_kwargs=carbon_kwargs,
        use_advanced_chemistry=use_advanced_chemistry,
        exotic_strength_enabled=exotic_strength_enabled,
        robust=robust_mode, design_age=design_age, ticket_config=ticket_config,
    )
