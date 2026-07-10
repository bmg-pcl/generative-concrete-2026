"""Compare Mixes tab — two mixes side by side with predicted performance."""
import streamlit as st

from src.exotics import EXOTIC_ADMIXTURES
from src.ui_logic import PARAM_NAMES, compute_metrics, mix_ticket
from ui.context import AppContext
from ui.state import SLIDER_SPECS, current_mix, load_mix_into

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

    def get_metrics(mix, exotic):
        return compute_metrics(
            mix, exotic, st.session_state.costs, ctx.predictor,
            advanced=ctx.use_advanced_chemistry,
            exotic_strength=ctx.exotic_strength_enabled,
            uncertainty_fn=ctx.bayesian.evaluate_uncertainty,
            carbon_kwargs=ctx.carbon_kwargs,
        )

    m_a = get_metrics(mix_a, st.session_state.exotic_a)
    m_b = get_metrics(mix_b, st.session_state.exotic_b)

    def strength_caption(metrics):
        """Honest note on how exotics relate to the strength number shown."""
        if ctx.exotic_strength_enabled and metrics.get("exotic_strength"):
            st.caption(f"includes {metrics['exotic_strength']:+.1f} MPa exotic estimate (unvalidated)")
        else:
            st.caption("model strength — exotics affect cost & carbon only")

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
        st.download_button("Download ticket (A)", key="ticket_a",
                           data=mix_ticket(dict(zip(param_names, mix_a)), m_a, ctx.ticket_config),
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
        st.download_button("Download ticket (B)", key="ticket_b",
                           data=mix_ticket(dict(zip(param_names, mix_b)), m_b, ctx.ticket_config),
                           file_name="mix_B_ticket.csv", mime="text/csv")
