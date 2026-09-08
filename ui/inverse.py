"""Inverse Design tab — recipes for a target strength, plus a design-space spread."""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui_logic import PARAM_NAMES, batch_metrics, recommend_recipe, mix_ticket, compute_metrics, carbon_term
from src.generative_ga import SLUMP_SP_DOSING_NOTE
from src.properties import get_slump_model
from ui.context import AppContext
from ui.state import load_mix_into, EXPOSURE_NONE, SLIDER_SPECS

param_names = list(PARAM_NAMES)


def _slump_model_available() -> bool:
    """R8.5 P3: the slump-target control is only meaningful when the trained
    PropertyModel artifact actually loads -- `get_slump_model()` raises
    FileNotFoundError (never a silent None) when it's missing, so this is a
    try/except probe, not a happy-path assumption."""
    try:
        get_slump_model()
        return True
    except Exception:
        return False


def _pick_compliant(checked, allow_unknown: bool):
    """R8.5 P4: apply OUR OWN strict-vs-allow-UNKNOWN acceptance criterion over
    `design_compliant()`'s `"checked"` list (every ranked candidate, best-first,
    with its real `check_compliance()` verdict -- `design_compliant` itself only
    ever accepts a true PASS). Strict (default): first verdict == "PASS".
    Allow-unknown: first verdict in {"PASS", "UNKNOWN"} -- "UNKNOWN" already
    means zero FAILing rules (check_compliance's own aggregation: any FAIL wins
    over UNKNOWN), so this is exactly "every EVALUABLE rule passes, some rules
    just aren't sourced by the pack." Never invents a laxer verdict than
    check_compliance itself computed -- it only changes which of its real,
    already-computed verdicts we're willing to accept."""
    for mix, result in checked:
        if result["verdict"] == "PASS":
            return mix, result
        if allow_unknown and result["verdict"] == "UNKNOWN":
            return mix, result
    return None, None


def render_inverse(ctx: AppContext):
    bayesian = ctx.bayesian
    predictor = ctx.predictor
    use_advanced_chemistry = ctx.use_advanced_chemistry
    carbon_kwargs = ctx.carbon_kwargs
    robust_mode = ctx.robust
    design_age = ctx.design_age

    st.header("Inverse Design: Recipes for a Target Strength")

    # Which generative backends are actually available right now?
    flow_ready = bayesian.amortized is not None
    backend_labels = {
        "auto": f"Auto ({'trained flow' if flow_ready else 'GA'})",
        "flow": "Amortized BayesFlow flow" + ("" if flow_ready else " — not trained"),
        "ga": "Genetic Algorithm (GA)",
        "aco": "Ant Colony (ACO)",
    }
    st.markdown(
        "Search — within the training-data envelope — for mixes whose model-predicted strength "
        "matches your target. Pick a **generative backend**: a trained amortized BayesFlow flow "
        "(instant, calibrated) when available, or a transparent metaheuristic (GA / ACO). "
        "See `docs/AMORTIZED_INFERENCE.md`."
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        target_str = st.number_input("Target Strength (MPa)", 10, 100, 45)
        backend = st.selectbox(
            "Generative backend", list(backend_labels.keys()),
            format_func=lambda k: backend_labels[k],
        )
    with c2:
        px_idx = st.selectbox("X-Axis Parameter", range(8), index=0, format_func=lambda x: param_names[x])
        py_idx = st.selectbox("Y-Axis Parameter", range(8), index=3, format_func=lambda x: param_names[x])

    if backend == "flow" and not flow_ready:
        st.warning("No trained flow weights found. Train with `python -m src.amortized`, "
                   "or choose GA / ACO. Falling back to GA for now.")
        backend = "ga"

    # --- R8.5 P2/P3/P4: optimizer-capability options ---------------------------
    st.subheader("Optimizer options")
    o1, o2, o3 = st.columns(3)
    with o1:
        robust_carbon = st.toggle(
            "Robust carbon (optimize the guaranteed +95% bound)",
            key="cfg_robust_carbon",
            help="ON: the reported carbon figure is the +1.96σ UPPER bound of the "
                 "per-material uncertainty band, not the point total -- carbon's "
                 "mirror of robust strength (which optimizes the LOWER bound). "
                 "Rewards well-characterized materials: attaching a supplier EPD "
                 "that tightens a factor's uncertainty genuinely lowers this "
                 "number. Disclosure/selection-figure swap only -- dimensionality "
                 "and the search itself are unaffected. OFF (default): point total.",
        )
    with o2:
        slump_available = _slump_model_available()
        slump_enabled = st.checkbox(
            "Enable workability (slump) target", key="cfg_slump_target_enabled",
            disabled=not slump_available,
            help=(
                "ON: bias the search toward a slump target, honestly gated by the "
                "SLUMP model's OWN support envelope (R8.1 -- distinct from, and "
                "narrower than, the strength envelope every other gate here uses). "
                "A target reachable only outside slump support is reported "
                "honestly, never a confident number."
                if slump_available else
                "The trained slump model artifact is not available in this "
                "environment (run `python -m src.properties`)."
            ),
        )
        slump_target_cm = None
        if slump_enabled and slump_available:
            slump_target_cm = st.number_input(
                "Target slump (cm)", 0.0, 29.0, key="cfg_slump_target_cm",
                help="Every slump-corpus row used superplasticizer ≥ 4.4 kg/m³ "
                     "(R8.1 WP-1b), so a slump-constrained design will be SP-dosed "
                     "-- that is the corpus speaking, not a preference.",
            )
    with o3:
        exposure_pack_id = st.session_state.get("cfg_exposure_pack", EXPOSURE_NONE)
        exposure_class_id = st.session_state.get("cfg_exposure_class", EXPOSURE_NONE)
        exposure_selected = (exposure_pack_id != EXPOSURE_NONE
                             and exposure_class_id != EXPOSURE_NONE)
        require_compliance = st.toggle(
            "Require compliance (selected exposure class)",
            key="cfg_require_compliance", disabled=not exposure_selected,
            help=(
                f"ON: only report a mix that VERIFIES compliant against "
                f"{exposure_pack_id}.{exposure_class_id} (the Config tab's "
                "selection) -- never a soft-biased-but-unverified candidate. "
                "GA/ACO backends only (the amortized-flow backend has no "
                "compliance-aware search)."
                if exposure_selected else
                "Select an exposure pack AND class on the Config tab to enable "
                "this."
            ),
        )
        allow_unknown = False
        if require_compliance and exposure_selected:
            allow_unknown = st.checkbox(
                "Allow UNKNOWN rules", key="cfg_compliance_allow_unknown",
                help="OFF (strict, default): every rule must verify PASS -- a "
                     "rule the selected pack does not source (UNKNOWN) counts as "
                     "a violation. ON: accept a design whose EVALUABLE rules all "
                     "PASS even when some rules are UNKNOWN (unsourced by the "
                     "pack) -- an explicit relaxation, disclosed wherever the "
                     "result is.",
            )
    if not slump_available:
        st.caption("Slump target unavailable: the trained slump model artifact was not found.")
    if require_compliance and exposure_selected and not allow_unknown:
        st.info(
            f"**Strict compliance mode.** {exposure_pack_id}.{exposure_class_id} "
            "omits one or more deemed-to-satisfy rules this model cannot source "
            "(e.g. the SCM cap) -- an omitted rule is UNKNOWN, and strict mode "
            "counts UNKNOWN as a violation (since minimizing carbon maximizes "
            "SCM, and the missing rule is exactly the SCM cap). A verified PASS "
            "may be structurally unreachable until the pack's sourcing improves "
            "-- a report of \"no compliant design found\" below means this "
            "honesty rule is working, not that anything is broken. Check "
            "\"Allow UNKNOWN rules\" above to see the best design against only "
            "this pack's SOURCED rules."
        )

    slump_target_value = float(slump_target_cm) if (slump_enabled and slump_available) else None
    compliance_pair = (exposure_pack_id, exposure_class_id) \
        if (require_compliance and exposure_selected) else None

    # Cache the (stochastic) sample cloud so unrelated reruns don't re-search.
    @st.cache_data(show_spinner="Sampling the design space…")
    def cached_samples(target, backend_key, n_samples, robust, age):
        return bayesian.sample_posterior(target, n_samples=n_samples, method=backend_key,
                                         robust=robust, age=age)

    samples = cached_samples(float(target_str), backend, 3000, robust_mode, design_age)
    # R7.2: the flow conditions on (strength, age), so a pinned age uses the flow too.
    used_backend = "trained flow" if (backend in ("auto", "flow") and flow_ready) else \
                   ("GA" if backend in ("auto", "ga") else "ACO")
    st.caption(f"Backend used: **{used_backend}** · {len(samples)} candidates sampled"
               + (f" · age conditioned to {design_age:.0f} d" if design_age is not None else ""))

    # --- Recommended recipe for the target -------------------------------------
    st.subheader("Recommended recipe")
    rec = None
    if compliance_pair is not None:
        # R8.5 P4: `recommend_recipe` (frozen, WP-1) has no `compliance` kwarg --
        # it never went through WP-3b's compliance machinery. Go straight to
        # `design_compliant()` on the metaheuristic designer (the engine's own
        # honest-verification wrapper: it re-checks the final ranked population
        # with the REAL check_compliance() engine, never presenting the search's
        # soft bias as a certificate), then apply our OWN strict/allow-unknown
        # acceptance criterion over its `"checked"` list (see `_pick_compliant`),
        # and package the chosen mix the same way `recommend_recipe` does, via
        # `compute_metrics` (which already computes slump/compliance/thermal
        # disclosure identically to every other surface -- see its own R8.5 P1
        # coherence-contract note).
        pack_id, cls = compliance_pair
        if backend not in ("ga", "aco"):
            st.info("Compliance-constrained search uses the GA backend (the "
                    "amortized-flow backend has no compliance-aware search).")
        compliant_backend = "aco" if backend == "aco" else "ga"
        designer = bayesian.aco_designer if compliant_backend == "aco" else bayesian.designer
        result = designer.design_compliant(
            float(target_str), compliance=compliance_pair,
            robust=robust_mode, age=design_age, slump_target=slump_target_value,
        )
        mix_d, check_result = _pick_compliant(result["checked"], allow_unknown)
        if mix_d is None:
            st.error(
                f"No design found whose {'evaluable rules all pass' if allow_unknown else 'rules verify PASS'} "
                f"against {pack_id}.{cls} within the search budget."
            )
        else:
            arr = np.array([mix_d[p] for p in param_names])
            m = compute_metrics(arr, {}, st.session_state.costs, predictor,
                                advanced=use_advanced_chemistry, carbon_kwargs=carbon_kwargs,
                                exposure_pack=pack_id, exposure_class=cls)
            carbon_disp = carbon_term(mix_d, use_advanced_chemistry, carbon_kwargs,
                                      robust_carbon=robust_carbon)
            rec = {
                "mix": arr, "params": mix_d, "strength": m["strength"],
                "interval_lo": m["interval_lo"], "interval_hi": m["interval_hi"],
                "novelty": m["novelty"], "in_support": m["in_support"],
                "workability": m["workability"], "tensile": m["tensile"], "curing": m["curing"],
                "carbon": carbon_disp, "carbon_basis": "upper_95" if robust_carbon else "point",
                "cost": m["cost"], "delta_t_adiabatic_C": m["delta_t_adiabatic_C"],
                "mass_pour_flag": m["mass_pour_flag"], "compliance": m["compliance"],
                "compliance_mode": "allow_unknown" if allow_unknown else "strict",
            }
            if slump_target_value is not None:
                rec["slump_target"] = slump_target_value
                rec["slump_cm"] = m["slump_cm"]
                rec["slump_lo"] = m["slump_lo"]
                rec["slump_hi"] = m["slump_hi"]
                rec["slump_basis"] = m["slump_basis"]
                rec["slump_in_support"] = m["slump_in_support"]
                rec["slump_reason"] = m["slump_reason"]
                rec["found"] = bool(m["slump_in_support"])
                if rec["found"]:
                    rec["slump_note"] = SLUMP_SP_DOSING_NOTE
                else:
                    st.warning(
                        "This compliant design sits outside the slump model's "
                        "support for the requested target — workability figure "
                        "withheld (basis: heuristic)."
                    )
    else:
        # Cached so it isn't re-searched on every unrelated rerun (it runs its
        # own backend search, separate from cached_samples above).
        @st.cache_data(show_spinner=False)
        def cached_recipe(target, backend_key, advanced, cost_items, carbon_key, robust, age,
                          robust_carbon_, slump_target_):
            transport_km_, cement_type_, factor_items, clinker_json, transport_detail_ = carbon_key
            ck = {"transport_km": transport_km_, "cement_type": cement_type_,
                  "factors": dict(factor_items), "clinker_source": json.loads(clinker_json),
                  "transport_detail": transport_detail_}
            return recommend_recipe(bayesian, target, method=backend_key, advanced=advanced,
                                    costs=dict(cost_items), carbon_kwargs=ck, robust=robust, age=age,
                                    robust_carbon=robust_carbon_, slump_target=slump_target_)

        rec = cached_recipe(float(target_str), backend, use_advanced_chemistry,
                            tuple(sorted(st.session_state.costs.items())),
                            (carbon_kwargs["transport_km"], carbon_kwargs["cement_type"],
                             tuple(sorted(carbon_kwargs["factors"].items())),
                             json.dumps(carbon_kwargs.get("clinker_source"), sort_keys=True),
                             # R8.5 P1: MUST be in the cache key, not just carbon_kwargs --
                             # without it, st.cache_data returns a stale result when only
                             # this toggle changes (same target/km/factors/clinker_source
                             # hash to the same cache entry).
                             bool(carbon_kwargs.get("transport_detail", False))),
                            robust_mode, design_age, robust_carbon, slump_target_value)

    if rec is not None:
        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted Strength", f"{rec['strength']:.1f} MPa", delta=f"{rec['strength']-target_str:+.1f} vs target")
        r2.metric("Carbon", f"{rec['carbon']:.1f} kg CO₂/m³",
                  help="+1.96σ upper bound (robust carbon)" if rec.get("carbon_basis") == "upper_95" else "point total")
        r3.metric("Nominal cost", f"${rec['cost']:.2f}/m³")
        st.caption(f"90% interval [{rec['interval_lo']:.0f}–{rec['interval_hi']:.0f}] MPa"
                   + (" · robust: optimized the guaranteed lower bound, kept in-support" if robust_mode else "")
                   + (" · carbon basis: upper_95 (robust)" if rec.get("carbon_basis") == "upper_95" else ""))
        # R8.6 WP-U4: robust mode matches the search to the *guaranteed lower
        # bound* (interval_lo), not the point estimate (see recommend_recipe's
        # docstring) -- so whenever that leaves the point estimate above the
        # target, it is the designed consequence of robust matching, not an
        # overshoot/error. Only shown when robust is on AND it actually
        # happened for this recipe; robust-off is unchanged (deliverable 2).
        if robust_mode and rec["strength"] > target_str:
            st.caption(
                "Robust mode matches the **guaranteed lower bound** to your target, "
                "not the point estimate above — the lower bound is what met "
                f"{target_str} MPa; the point estimate reads higher by design "
                "(the width of the uncertainty interval above that guaranteed floor), "
                "not an overshoot."
            )
        if not rec["in_support"]:
            st.warning("This recipe sits outside the well-sampled data region — the prediction is "
                       "extrapolated. Prefer a mix inside the data, or collect lab data here.")

        # R8.6 WP-U4: recipe as a compact table (kg/m³ per material + age),
        # replacing the old run-on caption text line -- and the Load buttons
        # directly under it, side by side.
        st.markdown("**Recipe (kg/m³, age in days)**")
        recipe_rows = [
            {"Material": label, "Amount": f"{rec['params'][p]:.0f} " + ("days" if p == "age" else "kg/m³")}
            for p, label, _lo, _hi in SLIDER_SPECS
        ]
        st.dataframe(pd.DataFrame(recipe_rows), hide_index=True, use_container_width=True)
        rec_vec = [rec["params"][p] for p in param_names]
        load_a, load_b = st.columns(2)
        # on_click callbacks write the keyed sliders BEFORE the Compare tab reinstantiates
        # them on the next run — the only clean way to set a keyed widget programmatically.
        load_a.button("Load into Mix A", key="load_rec_a",
                      on_click=load_mix_into, args=("A", rec_vec))
        load_b.button("Load into Mix B", key="load_rec_b",
                      on_click=load_mix_into, args=("B", rec_vec))

        if rec.get("workability"):
            st.caption(f"Workability: {rec['workability']}")
        # R8.5 P3: slump-target disclosure (recommend_recipe's own result already
        # carries these keys when a slump target was set; the compliance-path
        # branch above populates the identical shape).
        if rec.get("slump_target") is not None:
            if rec.get("slump_basis") == "model" and rec.get("slump_cm") is not None:
                st.caption(
                    f"Slump target {rec['slump_target']:.1f} cm → reached "
                    f"~{rec['slump_cm']:.1f} cm (basis: model, in slump support)."
                )
            else:
                st.caption(
                    f"Slump target {rec['slump_target']:.1f} cm: no in-support "
                    f"design found (basis: heuristic) — {rec.get('slump_reason') or 'outside the slump corpus.'}"
                )
            if rec.get("slump_note"):
                st.caption(rec["slump_note"])
        # R8.5 P5: post-hoc thermal advisory, same UNCALIBRATED status as the
        # Compare tab's other advisories -- never a constraint, only shown.
        if rec.get("delta_t_adiabatic_C") is not None:
            st.caption(f"Adiabatic ΔT ~{rec['delta_t_adiabatic_C']:.1f} °C "
                       "(uncalibrated planning signal, advisory only)")
        if rec.get("mass_pour_flag"):
            st.warning(rec["mass_pour_flag"])
        # R8.5 P4: compliance verdict + the mode that produced it.
        c = rec.get("compliance")
        if c:
            label = f"Compliance: {c['pack_id']}.{c['class']} — {c['verdict']} " \
                    f"(mode: {rec.get('compliance_mode', 'strict')})"
            if c["verdict"] == "PASS":
                st.success(label)
            elif c["verdict"] == "FAIL":
                st.error(label)
            else:
                st.warning(label)
        ticket_csv = mix_ticket(rec["params"], rec, ctx.ticket_config)
        if compliance_pair is not None:
            # mix_ticket (frozen) has no row for this -- append it ourselves in
            # the same "section,key,value" shape as its own `config,*` rows,
            # never reordering anything mix_ticket already wrote.
            ticket_csv += f"\nconfig,compliance_mode,{rec['compliance_mode']}"
        st.download_button(
            "Download recipe ticket (CSV)", key="ticket_rec",
            data=ticket_csv,
            file_name="recommended_recipe_ticket.csv", mime="text/csv",
        )

    # --- Density surface over two chosen parameters -----------------------------
    st.subheader("Design-space spread")
    hov = batch_metrics(samples[:300], st.session_state.costs, predictor,
                        advanced=use_advanced_chemistry, carbon_kwargs=carbon_kwargs)
    hover_texts = []
    for i in range(len(samples[:300])):
        lines = [
            f"<b>STRENGTH: {hov['strength'][i]:.1f} MPa</b>",
            f"<b>CARBON: {hov['carbon'][i]:.1f} kg/m³</b>",
            f"<b>COST: ${hov['cost'][i]:.2f}/m³</b>",
            "---",
        ]
        lines.extend([f"{p}: {samples[i, j]:.1f}" for j, p in enumerate(param_names)])
        hover_texts.append("<br>".join(lines))

    from scipy.stats import gaussian_kde
    x, y = samples[:, px_idx], samples[:, py_idx]
    fig = go.Figure()
    # gaussian_kde fails on a near-constant axis (singular covariance) — guard it.
    if np.std(x) < 1e-6 or np.std(y) < 1e-6:
        st.info("One of the chosen parameters is essentially constant for this target, "
                "so a density surface isn't meaningful — showing the raw sample scatter.")
        fig.add_trace(go.Scatter3d(
            x=x[:300], y=y[:300], z=hov["strength"], mode="markers",
            marker=dict(size=3, color=hov["strength"], colorscale="Magma", opacity=0.6),
            text=hover_texts, hoverinfo="text", name="Sample Recipes",
        ))
        z_title = "Predicted Strength (MPa)"
    else:
        kde = gaussian_kde(np.vstack([x, y]))
        xi, yi = np.mgrid[x.min():x.max():50j, y.min():y.max():50j]
        zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
        fig.add_trace(go.Surface(z=zi, x=xi, y=yi, colorscale="Magma", opacity=0.8,
                                 name="Density", showscale=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter3d(
            x=x[:300], y=y[:300], z=kde(np.vstack([x[:300], y[:300]])),
            mode="markers", marker=dict(size=3, color="cyan", opacity=0.5),
            text=hover_texts, hoverinfo="text", name="Sample Recipes",
        ))
        z_title = "Sample Density"

    fig.update_layout(
        template="plotly_dark",
        scene=dict(xaxis_title=param_names[px_idx].title(),
                   yaxis_title=param_names[py_idx].title(), zaxis_title=z_title),
        margin=dict(l=0, r=0, b=0, t=0), height=700,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="footnote">
    <strong>What you're seeing:</strong> a spread of mix designs whose predicted strength matches
    your target, produced by the selected backend. The <strong>trained amortized flow</strong>
    (BayesFlow normalizing flow) learns the inverse map p(mix | strength) once and then samples any
    target instantly — and is checked with Simulation-Based Calibration. The <strong>GA</strong> and
    <strong>ACO</strong> backends are transparent metaheuristics that search the same envelope with
    no neural network. The surface is a kernel-density estimate of the sampled recipes over the two
    chosen parameters. See <code>docs/AMORTIZED_INFERENCE.md</code> for the full explanation.
    </div>
    """, unsafe_allow_html=True)
