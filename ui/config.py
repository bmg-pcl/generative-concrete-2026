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
            help="One-way haul distance, plant to site; the 0.1 kg CO₂/t·km factor is "
                 "applied to the distance entered (not doubled for a return leg).",
        )
        # R8.0 WP-E Decision 2: DEFAULT OFF. The registry carries a default
        # transport block (mode + km) for most materials; applying it unconditionally
        # would silently add carbon to every existing mix that left "Transport
        # distance" at its 0 km default. ON swaps in per-material registry
        # distances for materials that carry a block, keeping the global-km field
        # above for the few that don't (water, superplasticizer).
        transport_detail = st.toggle(
            "Per-material transport detail (experimental)",
            key="cfg_transport_detail",
            help="OFF (default): the single 'Transport distance' above applies to "
                 "every material's mass. ON: materials with a registry transport "
                 "block (typical haul mode + distance, e.g. cement 200 km truck, "
                 "aggregates 30 km truck, exotics 500 km truck / 10,000 km ship for "
                 "nano materials) use THAT distance instead; only materials without "
                 "a registry block still use the global km above. This changes the "
                 "displayed carbon total for a dosed/default mix — off by default so "
                 "no existing number moves silently.",
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

        # R8.0 WP-A A2: batched material isn't all placed. Applied at the metrics/
        # ticket layer only (compute_metrics/mix_ticket) -- it never touches the
        # carbon_kwargs above, so the batched, per-m3 figures the optimizers target
        # are untouched. Keyed-only (no value=), same pattern as the other cfg_
        # controls.
        waste_factor = st.slider(
            "Waste factor (batched → placed)", 0.0, 0.25, step=0.01,
            key="cfg_waste_factor",
            help="Batched material is not all placed: spillage, over-ordering, and "
                 "pump/washout losses typically add 3-8% before compaction. This "
                 "scales the CARBON and COST 'as-placed' figures shown alongside the "
                 "batched, per-m³ numbers -- it does NOT change the mix design itself, "
                 "and the optimizers always target the batched (unscaled) figures.",
        )

        # R8.0 WP-E Decision 1: feeds ONLY the secondary `curing_maturity_days`
        # disclosure metric (src/thermal.py's ASTM C1074 maturity function) — it
        # never touches the primary `curing` heuristic or any carbon/cost figure.
        site_temp_c = st.number_input(
            "Site curing temperature (°C, secondary maturity estimate)", 0.0, 45.0,
            key="cfg_site_temp_c",
            help="Feeds an UNCALIBRATED, secondary maturity-based curing estimate "
                 "(ASTM C1074 Arrhenius) shown alongside — not instead of — the "
                 "primary curing heuristic above. The two estimate different "
                 "quantities and commonly disagree; that disagreement is itself "
                 "information, not a bug (see the ticket's "
                 "curing_maturity_days_uncalibrated row).",
        )

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
        "robust": robust_mode, "waste_factor": float(waste_factor),
        # R8.0 WP-E Decisions 1 & 2: UI-session concerns only (the CLI's run
        # config deliberately gains neither key — see src/cli.py), so they ride
        # only on the app's own ticket_config, not carbon_kwargs (carbon_kwargs
        # feeds every carbon call site, including the optimizers, which must keep
        # targeting today's batched figures unchanged).
        "transport_detail": bool(transport_detail),
        "site_temp_c": float(site_temp_c),
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
