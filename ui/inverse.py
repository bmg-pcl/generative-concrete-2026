"""Inverse Design tab — recipes for a target strength, plus a design-space spread."""
import json

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.ui_logic import PARAM_NAMES, batch_metrics, recommend_recipe, mix_ticket
from ui.context import AppContext
from ui.state import load_mix_into

param_names = list(PARAM_NAMES)


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

    # Cache the (stochastic) sample cloud so unrelated reruns don't re-search.
    @st.cache_data(show_spinner="Sampling the design space…")
    def cached_samples(target, backend_key, n_samples, robust, age):
        return bayesian.sample_posterior(target, n_samples=n_samples, method=backend_key,
                                         robust=robust, age=age)

    samples = cached_samples(float(target_str), backend, 3000, robust_mode, design_age)
    # A pinned design age routes away from the flow (it can't honor a fixed age).
    flow_usable = flow_ready and (design_age is None)
    used_backend = "trained flow" if (backend in ("auto", "flow") and flow_usable) else \
                   ("GA" if backend in ("auto", "ga") else "ACO")
    st.caption(f"Backend used: **{used_backend}** · {len(samples)} candidates sampled"
               + (f" · age pinned to {design_age:.0f} d" if design_age is not None else ""))
    if design_age is not None and backend in ("auto", "flow") and flow_ready:
        st.caption("Age is pinned, so the trained flow is bypassed for the GA designer "
                   "(the flow can't honor a fixed age — see roadmap R2.1).")

    # --- Recommended recipe for the target -------------------------------------
    # Cached so it isn't re-searched on every unrelated rerun (it runs its own
    # backend search, separate from cached_samples above).
    @st.cache_data(show_spinner=False)
    def cached_recipe(target, backend_key, advanced, cost_items, carbon_key, robust, age):
        transport_km_, cement_type_, factor_items, clinker_json = carbon_key
        ck = {"transport_km": transport_km_, "cement_type": cement_type_,
              "factors": dict(factor_items), "clinker_source": json.loads(clinker_json)}
        return recommend_recipe(bayesian, target, method=backend_key, advanced=advanced,
                                costs=dict(cost_items), carbon_kwargs=ck, robust=robust, age=age)

    rec = cached_recipe(float(target_str), backend, use_advanced_chemistry,
                        tuple(sorted(st.session_state.costs.items())),
                        (carbon_kwargs["transport_km"], carbon_kwargs["cement_type"],
                         tuple(sorted(carbon_kwargs["factors"].items())),
                         json.dumps(carbon_kwargs.get("clinker_source"), sort_keys=True)),
                        robust_mode, design_age)
    st.subheader("Recommended recipe")
    r1, r2, r3 = st.columns(3)
    r1.metric("Predicted Strength", f"{rec['strength']:.1f} MPa", delta=f"{rec['strength']-target_str:+.1f} vs target")
    r2.metric("Carbon", f"{rec['carbon']:.1f} kg CO₂/m³")
    r3.metric("Cost", f"${rec['cost']:.2f}/m³")
    st.caption(f"90% interval [{rec['interval_lo']:.0f}–{rec['interval_hi']:.0f}] MPa"
               + (" · robust: optimized the guaranteed lower bound, kept in-support" if robust_mode else ""))
    if not rec["in_support"]:
        st.warning("This recipe sits outside the well-sampled data region — the prediction is "
                   "extrapolated. Prefer a mix inside the data, or collect lab data here.")
    if rec.get("workability"):
        st.caption(f"Workability: {rec['workability']}")
    st.caption("  ·  ".join(f"{p}: {rec['params'][p]:.0f}" for p in param_names))
    rec_vec = [rec["params"][p] for p in param_names]
    load_a, load_b = st.columns(2)
    # on_click callbacks write the keyed sliders BEFORE the Compare tab reinstantiates
    # them on the next run — the only clean way to set a keyed widget programmatically.
    load_a.button("Load into Mix A", key="load_rec_a",
                  on_click=load_mix_into, args=("A", rec_vec))
    load_b.button("Load into Mix B", key="load_rec_b",
                  on_click=load_mix_into, args=("B", rec_vec))
    st.download_button(
        "Download recipe ticket (CSV)", key="ticket_rec",
        data=mix_ticket(rec["params"], rec, ctx.ticket_config),
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
