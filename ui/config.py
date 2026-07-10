"""Config tab — costing, carbon model, and experimental options.

`render_config()` renders the tab and returns the AppContext the other tabs consume;
building the context here is what enforces "Config is read before the other tabs".
"""
import json
from datetime import datetime, timezone

import streamlit as st

from src.exotics import EXOTIC_STRENGTH_DISCLAIMER
from src.materials import validate_epd_json, carbon_provenance
from src.chemistry_advanced import FUEL_EF, GRID_EF, clinker_scope_split
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
        # EPD upload sits ABOVE the factor editors and is processed first: attaching
        # an EPD writes the cf_ widget keys, which is only legal before those widgets
        # instantiate on this run (the R4.1 keyed-widget rule).
        with st.expander("Supplier EPDs (plug in measured factors)"):
            st.caption(
                "Attach supplier-specific EPD values as JSON: "
                '`{"epds": {"cement": {"value": 0.55, "reference": "EPD-XYZ-2025"}}}`. '
                "An attached EPD replaces the database default below — measured beats "
                "modeled. Editing a factor afterwards records it as a user override. "
                "The mix ticket discloses which source produced each factor."
            )
            epd_file = st.file_uploader("EPD file (JSON)", type="json", key="epd_upload")
            if epd_file is not None:
                try:
                    epd_data = json.load(epd_file)
                except json.JSONDecodeError:
                    epd_data = None
                epd_err = validate_epd_json(epd_data) if epd_data is not None \
                    else "File is not valid JSON."
                if epd_err:
                    st.error(f"EPD import failed: {epd_err}")
                else:
                    # Apply once per distinct upload, so later manual factor edits
                    # are not clobbered on every rerun while the file stays attached.
                    sig = json.dumps(epd_data["epds"], sort_keys=True)
                    if st.session_state.get("epds_applied") != sig:
                        st.session_state.epds = epd_data["epds"]
                        for mat, rec in epd_data["epds"].items():
                            if mat in st.session_state.carbon_factors:
                                st.session_state.carbon_factors[mat] = float(rec["value"])
                                st.session_state[f"cf_{mat}"] = float(rec["value"])
                        st.session_state.epds_applied = sig
                    st.success("EPDs attached for: " + ", ".join(epd_data["epds"]))
            if st.session_state.get("epds"):
                st.caption("EPD-backed factors: " + ", ".join(
                    f"{m} ({r.get('reference', 'unreferenced')})"
                    for m, r in st.session_state.epds.items()))

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
        # Keyed-only (no value=): the cfg_ key IS the live value, initialised in
        # init_session_state() and written by session import (R7.4a) — same
        # single-source pattern as the mix sliders and cf_<mat> factors (R4.1/R5.1).
        chemistry_mode = st.radio(
            "Carbon model",
            ["Simple (Linear)", "Advanced (Molecular)"],
            key="cfg_chemistry_mode",
            help="Simple uses mass x factor. Advanced uses Bogue calculations and clinker chemistry.",
        )
        use_advanced_chemistry = (chemistry_mode == "Advanced (Molecular)")

        transport_km = st.number_input(
            "Transport distance (km)", 0, 2000,
            key="cfg_transport_km",
            help="Round-trip haul from plant to site. Adds ~0.1 kg CO₂ per tonne per km.",
        )
        cement_source = st.selectbox(
            "Clinker / cement source", ["OPC (Portland)", "LC3 (limestone calcined clay)"],
            key="cfg_cement_source",
            help="Advanced tier only: sets the clinker factor (OPC ≈ 0.95, LC3 ≈ 0.50).",
        )
        cement_type = "LC3" if cement_source.startswith("LC3") else "OPC"

        with st.expander("Clinker source (advanced tier)"):
            st.caption(
                "Differentiate HOW the clinker was made: a carbon-captured plant, an "
                "electrified kiln on hydro, or a standard gas/coal kiln produce very "
                "different numbers from identical chemistry. Applies to the Advanced "
                "carbon model's cement term; a supplier EPD (left) beats this model."
            )
            fuel_choice = st.selectbox(
                "Kiln fuel", ["Default (unspecified mix)"] + sorted(FUEL_EF.keys()),
                key="cfg_kiln_fuel",
                help="'electric' moves kiln heat to the electricity source below (Scope 2).",
            )
            if fuel_choice.startswith("Default"):
                clinker_source = None
            else:
                electricity = st.selectbox(
                    "Electricity source", sorted(GRID_EF.keys()),
                    key="cfg_electricity",
                    help="Powers an electrified kiln and/or the capture plant. "
                         "Hydro or a clean PPA makes those terms nearly free.",
                )
                capture_rate = st.slider(
                    "Carbon capture rate", 0.0, 0.95, step=0.05,
                    key="cfg_capture_rate",
                    help="Fraction of stack CO₂ (calcination + combustion) captured. "
                         "Capture's energy penalty is charged to the electricity source.",
                )
                clinker_source = {"kiln_fuel": fuel_choice, "electricity": electricity}
                if capture_rate > 0:
                    clinker_source["capture"] = {"rate": float(capture_rate)}
                split = clinker_scope_split(1.0, clinker_source)
                st.caption(
                    f"Clinker: {split['total']:.2f} kg CO₂/kg "
                    f"(Scope 1 stack {split['scope1']:.2f} + Scope 2 electricity "
                    f"{split['scope2']:.2f}) — legacy default is 0.88."
                )

        # One carbon config threaded to every carbon computation across the tabs.
        carbon_kwargs = {
            "transport_km": float(transport_km),
            "cement_type": cement_type,
            "factors": st.session_state.carbon_factors,
            "clinker_source": clinker_source,
        }

        st.divider()
        st.subheader("Optimization")
        robust_mode = st.toggle(
            "Robust mode (optimize the guaranteed-strength bound, stay in-data)",
            key="cfg_robust",
            help="ON: inverse design and Pareto optimize the conformal LOWER bound of "
                 "strength and keep candidates inside the trusted data region "
                 "(NSGA enforces it as a constraint). OFF: optimize the mean prediction, "
                 "which can chase confident-but-unsupported optima.",
        )
        fix_age = st.checkbox(
            "Fix design age (inverse design & Pareto)", key="cfg_fix_age",
            help="ON: hold age fixed so the optimizer can't hit a strength target by "
                 "prescribing a long cure (age is a condition, not a design variable). "
                 "The trained flow conditions on (strength, age) since R7.2, so a "
                 "pinned age is honored by the flow directly. OFF: age is free.",
        )
        design_age = st.number_input("Design age (days)", 1, 365, key="cfg_design_age") \
            if fix_age else None

        st.divider()
        st.subheader("Exotic Strength Model")
        exotic_strength_enabled = st.toggle(
            "Include exotics in predicted strength (experimental)",
            key="cfg_exotic_strength",
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
        "carbon_provenance": carbon_provenance(st.session_state.carbon_factors,
                                               st.session_state.get("epds")),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return AppContext(
        predictor=predictor, bayesian=bayesian, presets=presets,
        costs=st.session_state.costs, carbon_kwargs=carbon_kwargs,
        use_advanced_chemistry=use_advanced_chemistry,
        exotic_strength_enabled=exotic_strength_enabled,
        robust=robust_mode, design_age=design_age, ticket_config=ticket_config,
    )
