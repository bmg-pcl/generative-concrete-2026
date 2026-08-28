"""Compare Mixes tab — two mixes side by side with predicted performance."""
import streamlit as st

from src.exotics import EXOTIC_ADMIXTURES, compliance_warnings
from src.ui_logic import (
    PARAM_NAMES,
    compute_metrics,
    mix_ticket,
    compliance_advisory_text,
    compliance_matrix,
    slump_caveat,
)
from ui.context import AppContext
from ui.state import SLIDER_SPECS, current_mix, load_mix_into, EXPOSURE_NONE

param_names = list(PARAM_NAMES)


def render_compare(ctx: AppContext):
    st.markdown("""
    **How to use:** Configure two mix designs side-by-side to compare predicted performance.
    Use the preset dropdown to load known mixtures, or adjust sliders manually.
    """)
    col_a, col_b = st.columns(2)

    def _apply_preset(slot):
        """on_change for the preset selectbox: load the preset into the keyed sliders."""
        mix = ctx.presets.get(st.session_state[f"preset_{slot}"])
        if mix is not None:
            load_mix_into(slot, mix)

    def mix_input_ui(slot, exotic_state):
        st.selectbox(f"Load Preset ({slot})", list(ctx.presets.keys()),
                     key=f"preset_{slot}", on_change=_apply_preset, args=(slot,))

        with st.expander(f"Standard Components ({slot})", expanded=True):
            col1, col2 = st.columns(2)
            cols = [col1] * 4 + [col2] * 4
            for (p, label, lo, hi), c in zip(SLIDER_SPECS, cols):
                with c:
                    # Keyed, no value= — Streamlit persists the value under the key.
                    st.slider(f"{label} ({slot})", lo, hi, key=f"{p}_{slot}")

        with st.expander(f"Exotic Admixtures ({slot})", expanded=False):
            if ctx.exotic_strength_enabled:
                st.caption("Strength effect ON (experimental, unvalidated estimate).")
            else:
                st.caption("Affect cost & carbon only. Enable the exotic strength model in the Config tab to include them in strength.")
            for adm, props in EXOTIC_ADMIXTURES.items():
                exotic_state[adm] = st.slider(f"{adm.replace('_', ' ').title()} ({slot})", 0, props["max"], exotic_state.get(adm, 0))

        return current_mix(slot), exotic_state

    with col_a:
        mix_a, st.session_state.exotic_a = mix_input_ui("A", st.session_state.exotic_a)
    with col_b:
        mix_b, st.session_state.exotic_b = mix_input_ui("B", st.session_state.exotic_b)

    # R8.2 WP-3: the exposure pack/class selectors live on the Config tab
    # (ui/config.py), keyed cfg_exposure_pack/cfg_exposure_class -- read via
    # session_state, same reason as cfg_waste_factor above (ui/context.py is
    # outside this work package's file ownership). EXPOSURE_NONE on either
    # selector maps to None here, which is what makes compute_metrics's
    # compliance block inert.
    exposure_pack_id = st.session_state.get("cfg_exposure_pack", EXPOSURE_NONE)
    exposure_class_id = st.session_state.get("cfg_exposure_class", EXPOSURE_NONE)
    if exposure_pack_id == EXPOSURE_NONE:
        exposure_pack_id = None
    if exposure_class_id == EXPOSURE_NONE:
        exposure_class_id = None

    def get_metrics(mix, exotic):
        return compute_metrics(
            mix, exotic, st.session_state.costs, ctx.predictor,
            advanced=ctx.use_advanced_chemistry,
            exotic_strength=ctx.exotic_strength_enabled,
            uncertainty_fn=ctx.bayesian.evaluate_uncertainty,
            carbon_kwargs=ctx.carbon_kwargs,
            # cfg_waste_factor/cfg_transport_detail/cfg_site_temp_c are keyed
            # Config-tab widgets (R8.0 WP-A A2 / WP-E Decisions 1 & 2); read via
            # session_state rather than AppContext (ui/context.py is outside this
            # work package's file ownership).
            waste_factor=float(st.session_state.get("cfg_waste_factor", 0.0)),
            transport_detail=bool(st.session_state.get("cfg_transport_detail", False)),
            site_temp_c=float(st.session_state.get("cfg_site_temp_c", 20.0)),
            exposure_pack=exposure_pack_id,
            exposure_class=exposure_class_id,
        )

    m_a = get_metrics(mix_a, st.session_state.exotic_a)
    m_b = get_metrics(mix_b, st.session_state.exotic_b)

    def strength_caption(metrics):
        """Honest note on how exotics relate to the strength number shown."""
        if ctx.exotic_strength_enabled and metrics.get("exotic_strength"):
            st.caption(f"includes {metrics['exotic_strength']:+.1f} MPa exotic estimate (unvalidated)")
        else:
            st.caption("model strength — exotics affect cost & carbon only")

    def slump_display(metrics):
        """R8.1 WP-1b's two non-negotiable display rules: show the POINT ESTIMATE,
        state the interval width plainly as a caveat, and NEVER render it as an
        error bar or a guaranteed range the way strength's interval is rendered.
        Out-of-support mixes show the heuristic result with its basis label, never
        a model number."""
        if metrics["slump_basis"] == "model":
            st.caption(f"Slump ~{metrics['slump_cm']:.1f} cm (point estimate, basis: model)")
            st.caption(slump_caveat(metrics["slump_lo"], metrics["slump_hi"]))
        else:
            st.caption(
                f"Slump: no measured estimate (basis: heuristic) — "
                f"{metrics.get('slump_reason') or 'outside the slump corpus.'}"
            )

    def render_compliance(metrics, slot):
        """R8.2's per-mix compliance panel: verdict banner, per-rule table, and
        the mandatory advisory disclosure. UNKNOWN renders via st.warning (amber),
        deliberately distinct from PASS's st.success (green) and FAIL's st.error
        (red) -- never collapsed together."""
        c = metrics.get("compliance")
        if not c:
            return
        label = f"Compliance ({slot}): {c['pack_id']}.{c['class']} — {c['verdict']}"
        if c["verdict"] == "PASS":
            st.success(label)
        elif c["verdict"] == "FAIL":
            st.error(label)
        else:  # UNKNOWN — never collapsed into PASS
            st.warning(label)
        st.caption(compliance_advisory_text(c.get("source", {})))
        if c["rules"]:
            st.dataframe(
                [{"rule": r["rule"], "required": r["required"], "actual": r["actual"],
                 "result": r["result"], "reason": r["reason"] or ""} for r in c["rules"]],
                hide_index=True, use_container_width=True,
            )

    st.divider()
    res_a, res_b = st.columns(2)
    with res_a:
        st.subheader("Mix A")
        st.metric("Strength", f"{m_a['strength']:.1f} MPa")
        strength_caption(m_a)
        st.metric("Carbon", f"{m_a['carbon']:.1f} kg CO₂/m³")
        st.metric("Cost", f"${m_a['cost']:.2f}/m³")
        st.caption(f"90% interval [{m_a['interval_lo']:.0f}–{m_a['interval_hi']:.0f}] MPa · "
                   f"tensile ~{m_a['tensile']:.1f} MPa (EC2 derived) · curing ~{m_a['curing']:.0f} d (heuristic)")
        if not m_a["in_support"]:
            st.warning("Outside the well-sampled data region — treat this prediction as extrapolation.")
        if m_a["workability"]:
            st.caption(f"Workability: {m_a['workability']}")
        slump_display(m_a)
        # R8.0 WP-D2/WP-E: compliance advisories for any dosed restricted material
        # (e.g. calcium_chloride in reinforced concrete) — advisory, not a hard
        # constraint; this tool doesn't know the end use, only that the question exists.
        for w in compliance_warnings(st.session_state.exotic_a):
            st.warning(w)
        render_compliance(m_a, "A")
        st.download_button("Download ticket (A)", key="ticket_a",
                           data=mix_ticket(dict(zip(param_names, mix_a)), m_a, ctx.ticket_config,
                                           exotic=st.session_state.exotic_a),
                           file_name="mix_A_ticket.csv", mime="text/csv")
    with res_b:
        st.subheader("Mix B")
        st.metric("Strength", f"{m_b['strength']:.1f} MPa", delta=f"{m_b['strength']-m_a['strength']:.1f}")
        strength_caption(m_b)
        st.metric("Carbon", f"{m_b['carbon']:.1f} kg CO₂/m³", delta=f"{m_b['carbon']-m_a['carbon']:.1f}", delta_color="inverse")
        st.metric("Cost", f"${m_b['cost']:.2f}/m³", delta=f"${m_b['cost']-m_a['cost']:.2f}", delta_color="inverse")
        st.caption(f"90% interval [{m_b['interval_lo']:.0f}–{m_b['interval_hi']:.0f}] MPa · "
                   f"tensile ~{m_b['tensile']:.1f} MPa (EC2 derived) · curing ~{m_b['curing']:.0f} d (heuristic)")
        if not m_b["in_support"]:
            st.warning("Outside the well-sampled data region — treat this prediction as extrapolation.")
        if m_b["workability"]:
            st.caption(f"Workability: {m_b['workability']}")
        slump_display(m_b)
        for w in compliance_warnings(st.session_state.exotic_b):
            st.warning(w)
        render_compliance(m_b, "B")
        st.download_button("Download ticket (B)", key="ticket_b",
                           data=mix_ticket(dict(zip(param_names, mix_b)), m_b, ctx.ticket_config,
                                           exotic=st.session_state.exotic_b),
                           file_name="mix_B_ticket.csv", mime="text/csv")

    # R8.2 WP-3: the cross-jurisdiction table -- the feature's headline, since it
    # makes national variation visible at a glance. Always rendered (not gated on
    # a pack being selected above): even with no primary pack/class chosen, this
    # shows both mixes against a representative class from every known
    # jurisdiction pack (compliance_matrix's default); selecting a pack/class
    # above swaps that pack's row for the user's own choice.
    st.divider()
    st.subheader("Cross-jurisdiction compliance (advisory)")
    st.caption(
        "The same two mixes checked against each jurisdiction pack's own "
        "representative exposure class (named in every row). **The classes are "
        "not equivalent requirements** — EN 206 and ACI 318 use different "
        "taxonomies, so a differing verdict reflects a difference in what was "
        "checked, not necessarily a regulatory difference. Pick a pack and class "
        "above to check one deliberately. Always advisory; check the named standard before any structural "
        "use (see each pack's own disclosure above)."
    )
    rows_a = compliance_matrix(
        dict(zip(param_names, mix_a)), strength_lo=m_a["interval_lo"],
        highlight_pack=exposure_pack_id, highlight_class=exposure_class_id,
    )
    rows_b = compliance_matrix(
        dict(zip(param_names, mix_b)), strength_lo=m_b["interval_lo"],
        highlight_pack=exposure_pack_id, highlight_class=exposure_class_id,
    )
    by_key_b = {(r["pack_id"], r["class"]): r for r in rows_b}
    table = []
    for ra in rows_a:
        rb = by_key_b.get((ra["pack_id"], ra["class"]))
        table.append({
            "jurisdiction": f"{ra['pack_id']}.{ra['class']}",
            "Mix A": ra["verdict"], "Mix B": rb["verdict"] if rb else "n/a",
        })
    if table:
        st.dataframe(table, hide_index=True, use_container_width=True)
