import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import re
from src.models import StrengthPredictor
from src.chemistry_simple import calculate_mix_cost, UNIT_COSTS
from src.bayesian import BayesFlowExplorer
from src.ga import GeneticOptimizer
from src.annealing import SimulatedAnnealing
from src.nsga import run_nsga, pymoo_available
from src.data_fetcher import append_experimental_results, load_data
from src.exotics import EXOTIC_ADMIXTURES, EXOTIC_STRENGTH_DISCLAIMER
from src.ui_logic import (
    PARAM_NAMES,
    compute_metrics,
    batch_metrics,
    scalarized_fitness,
    recommend_recipe,
    carbon_for_mode,
    pareto_front_mask,
    validate_lab_csv,
    validate_session_state,
)

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

# --- Loading Status ---
status_container = st.empty()
status_container.info("Initializing models...")

# --- Preset Mixtures from Dataset ---
@st.cache_data
def get_preset_mixtures():
    df = load_data()
    presets = {"Custom": None}
    # Label each sample with its ACTUAL measured strength from the dataset, rather
    # than asserting an unverified qualitative property.
    for row in (42, 100, 200, 500):
        strength = df.iloc[row]["strength"]
        presets[f"Dataset #{row} ({strength:.0f} MPa measured)"] = df.iloc[row].values[:8]
    return presets

# --- Initialize models ---
@st.cache_resource
def load_resources():
    predictor = StrengthPredictor()
    try:
        predictor.predict(np.zeros((1, 8))) 
    except:
        predictor.train()
    
    bayesian = BayesFlowExplorer()
    return predictor, bayesian

status_container.info("Loading strength predictor...")
predictor, bayesian = load_resources()

status_container.info("Loading dataset presets...")
PRESET_MIXTURES = get_preset_mixtures()

status_container.empty()  # Clear loading message

param_names = list(PARAM_NAMES)  # single source in src/generative_ga.py

# --- State Management ---
if 'mix_a' not in st.session_state:
    st.session_state.mix_a = np.array([300, 0, 0, 180, 0, 1000, 800, 28])
if 'mix_b' not in st.session_state:
    st.session_state.mix_b = np.array([300, 100, 50, 160, 5, 1000, 800, 28])
if 'costs' not in st.session_state:
    st.session_state.costs = UNIT_COSTS.copy()
if 'exotic_a' not in st.session_state:
    st.session_state.exotic_a = {k: v["default"] for k, v in EXOTIC_ADMIXTURES.items()}
if 'exotic_b' not in st.session_state:
    st.session_state.exotic_b = {k: v["default"] for k, v in EXOTIC_ADMIXTURES.items()}

# --- App Header ---
st.markdown('<h1 class="main-title">Generative Mix Design</h1>', unsafe_allow_html=True)
st.markdown("AI-powered concrete formulation: prediction, optimization, and inverse design.")

# --- Sidebar: step-by-step guidance + session tools ---
def get_state_json():
    state = {
        "mix_a": st.session_state.mix_a.tolist(),
        "mix_b": st.session_state.mix_b.tolist(),
        "costs": st.session_state.costs,
        "exotic_a": st.session_state.exotic_a,
        "exotic_b": st.session_state.exotic_b,
    }
    return json.dumps(state)

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
                st.session_state.mix_a = np.array(data["mix_a"])
                st.session_state.mix_b = np.array(data["mix_b"])
                st.session_state.costs = data["costs"]
                if "exotic_a" in data:
                    st.session_state.exotic_a = data["exotic_a"]
                    st.session_state.exotic_b = data["exotic_b"]
                st.success("Session Imported!")

# --- Main Layout ---
tab1, tab2, tab3, tab4, tab_config, tab_workflow, tab5, tab6 = st.tabs([
    "Compare Mixes", "Inverse Design", "Pareto Optimization",
    "Calibration", "Config", "Workflow", "Technical Report", "References",
])

# The Config tab holds costing and carbon settings. Its `with` block runs FIRST
# (before the other tabs) so use_advanced_chemistry / exotic_strength_enabled and
# the material costs are defined by the time the other tabs read them.
with tab_config:
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
    with cfg_model:
        st.subheader("Carbon & Analysis Model")
        chemistry_mode = st.radio(
            "Carbon model",
            ["Simple (Linear)", "Advanced (Molecular)"],
            help="Simple uses mass x factor. Advanced uses Bogue calculations and clinker chemistry.",
        )
        use_advanced_chemistry = (chemistry_mode == "Advanced (Molecular)")

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

with tab1:
    st.markdown("""
    **How to use:** Configure two mix designs side-by-side to compare predicted performance. 
    Use the preset dropdown to load known mixtures, or adjust sliders manually.
    """)
    col_a, col_b = st.columns(2)
    
    def mix_input_ui(key, default_mix, exotic_state):
        preset_choice = st.selectbox(f"Load Preset ({key})", list(PRESET_MIXTURES.keys()), key=f"preset_{key}")
        if preset_choice != "Custom" and PRESET_MIXTURES[preset_choice] is not None:
            default_mix = PRESET_MIXTURES[preset_choice].copy()
        
        with st.expander(f"Standard Components ({key})", expanded=True):
            col1, col2 = st.columns(2)
            mix_vals = np.array(default_mix).copy()
            with col1:
                mix_vals[0] = st.slider(f"Cement ({key})", 100, 550, int(mix_vals[0]))
                mix_vals[1] = st.slider(f"Slag ({key})", 0, 360, int(mix_vals[1]))
                mix_vals[2] = st.slider(f"Fly Ash ({key})", 0, 200, int(mix_vals[2]))
                mix_vals[3] = st.slider(f"Water ({key})", 120, 250, int(mix_vals[3]))
            with col2:
                mix_vals[4] = st.slider(f"Superplasticizer ({key})", 0, 30, int(mix_vals[4]))
                mix_vals[5] = st.slider(f"Coarse Agg ({key})", 700, 1150, int(mix_vals[5]))
                mix_vals[6] = st.slider(f"Fine Agg ({key})", 550, 1000, int(mix_vals[6]))
                mix_vals[7] = st.slider(f"Age (days) ({key})", 1, 365, int(mix_vals[7]))
        
        with st.expander(f"Exotic Admixtures ({key})", expanded=False):
            if exotic_strength_enabled:
                st.caption("Strength effect ON (experimental, unvalidated estimate).")
            else:
                st.caption("Affect cost & carbon only. Enable the exotic strength model in the Config tab to include them in strength.")
            for adm, props in EXOTIC_ADMIXTURES.items():
                exotic_state[adm] = st.slider(f"{adm.replace('_', ' ').title()} ({key})", 0, props["max"], exotic_state.get(adm, 0))
        
        return mix_vals, exotic_state

    with col_a:
        st.session_state.mix_a, st.session_state.exotic_a = mix_input_ui("A", st.session_state.mix_a, st.session_state.exotic_a)
    with col_b:
        st.session_state.mix_b, st.session_state.exotic_b = mix_input_ui("B", st.session_state.mix_b, st.session_state.exotic_b)

    def get_metrics(mix, exotic):
        return compute_metrics(
            mix, exotic, st.session_state.costs, predictor,
            advanced=use_advanced_chemistry,
            exotic_strength=exotic_strength_enabled,
            uncertainty_fn=bayesian.evaluate_uncertainty,
        )

    m_a = get_metrics(st.session_state.mix_a, st.session_state.exotic_a)
    m_b = get_metrics(st.session_state.mix_b, st.session_state.exotic_b)

    def strength_caption(metrics):
        """Honest note on how exotics relate to the strength number shown."""
        if exotic_strength_enabled and metrics.get("exotic_strength"):
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
        st.caption(f"model uncertainty ±{m_a['uncertainty']:.1f} MPa (heuristic) · curing ~{m_a['curing']:.0f} d")
    with res_b:
        st.subheader("Mix B")
        st.metric("Strength", f"{m_b['strength']:.1f} MPa", delta=f"{m_b['strength']-m_a['strength']:.1f}")
        strength_caption(m_b)
        st.metric("Carbon", f"{m_b['carbon']:.1f} kg CO₂/m³", delta=f"{m_b['carbon']-m_a['carbon']:.1f}", delta_color="inverse")
        st.metric("Cost", f"${m_b['cost']:.2f}/m³", delta=f"${m_b['cost']-m_a['cost']:.2f}", delta_color="inverse")
        st.caption(f"model uncertainty ±{m_b['uncertainty']:.1f} MPa (heuristic) · curing ~{m_b['curing']:.0f} d")

with tab2:
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
    def cached_samples(target, backend_key, n_samples):
        return bayesian.sample_posterior(target, n_samples=n_samples, method=backend_key)

    samples = cached_samples(float(target_str), backend, 3000)
    used_backend = "trained flow" if (backend in ("auto", "flow") and flow_ready) else \
                   ("GA" if backend in ("auto", "ga") else "ACO")
    st.caption(f"Backend used: **{used_backend}** · {len(samples)} candidates sampled")

    # --- Recommended recipe for the target -------------------------------------
    # Cached so it isn't re-searched on every unrelated rerun (it runs its own
    # backend search, separate from cached_samples above).
    @st.cache_data(show_spinner=False)
    def cached_recipe(target, backend_key, advanced, cost_items):
        return recommend_recipe(bayesian, target, method=backend_key,
                                advanced=advanced, costs=dict(cost_items))

    rec = cached_recipe(float(target_str), backend, use_advanced_chemistry,
                        tuple(sorted(st.session_state.costs.items())))
    st.subheader("Recommended recipe")
    r1, r2, r3 = st.columns(3)
    r1.metric("Predicted Strength", f"{rec['strength']:.1f} MPa", delta=f"{rec['strength']-target_str:+.1f} vs target")
    r2.metric("Carbon", f"{rec['carbon']:.1f} kg CO₂/m³")
    r3.metric("Cost", f"${rec['cost']:.2f}/m³")
    st.caption("  ·  ".join(f"{p}: {rec['params'][p]:.0f}" for p in param_names))
    load_a, load_b = st.columns(2)
    if load_a.button("Load into Mix A"):
        st.session_state.mix_a = np.array([rec["params"][p] for p in param_names])
        st.success("Loaded into Mix A — see the Compare Mixes tab.")
    if load_b.button("Load into Mix B"):
        st.session_state.mix_b = np.array([rec["params"][p] for p in param_names])
        st.success("Loaded into Mix B — see the Compare Mixes tab.")

    # --- Density surface over two chosen parameters -----------------------------
    st.subheader("Design-space spread")
    hov = batch_metrics(samples[:300], st.session_state.costs, predictor, advanced=use_advanced_chemistry)
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

with tab3:
    st.header("Multi-Objective Pareto Optimization")
    st.markdown("""
    **How to use:** Pick an algorithm and click Run. **GA / SA** search a *weighted* objective
    (you set the weights) and stream live; **NSGA-II / NSGA-III** are true multi-objective methods
    that map the whole **strength / carbon / cost** Pareto front in one run — no weights, and they
    can warm-start from the inverse-design flow. See the Workflow tab for when to use which.
    """)
    
    col_cfg, col_algo = st.columns([1, 1])
    with col_cfg:
        algo_options = ["Genetic Algorithm (GA)", "Simulated Annealing (SA)"]
        if pymoo_available():
            algo_options += ["NSGA-II (multi-objective)", "NSGA-III (multi-objective)"]
        algorithm = st.selectbox("Optimization Algorithm", algo_options)
        is_nsga = "NSGA" in algorithm

        if algorithm == "Genetic Algorithm (GA)":
            pop_size = st.number_input("Population Size", 20, 200, 50)
            n_gens = st.number_input("Generations", 10, 100, 30)
        elif algorithm == "Simulated Annealing (SA)":
            initial_temp = st.number_input("Initial Temperature", 100, 5000, 1000)
            cooling_rate = st.slider("Cooling Rate", 0.80, 0.99, 0.95)
            n_steps = st.number_input("Max Temperature Steps", 20, 200, 50)
        else:  # NSGA-II / NSGA-III
            nsga_pop = st.number_input("Population Size", 20, 200, 60, key="nsga_pop")
            nsga_gen = st.number_input("Generations", 10, 120, 40, key="nsga_gen")
            warm = st.checkbox(
                "Warm-start from inverse design at a target", value=False,
                help="Seed the initial population with realistic mixes from the flow/GA near a "
                     "target strength, so NSGA converges faster and stays in-distribution.",
            )
            warm_target = st.number_input("Warm-start target (MPa)", 10, 100, 45) if warm else None

        if is_nsga:
            st.caption("NSGA maps the whole strength / carbon / cost trade-off surface — no scalar weights needed.")
        else:
            st.subheader("Objective Weights")
            w_strength = st.slider("Strength Weight", 0.0, 2.0, 1.0, help="Weight for maximizing compressive strength.")
            w_carbon = st.slider("Carbon Penalty Weight", 0.0, 1.0, 0.05, help="Weight for minimizing embodied carbon.")
            w_cost = st.slider("Cost Penalty Weight", 0.0, 1.0, 0.5, help="Weight for minimizing material cost.")
    
    bounds = [(100, 550), (0, 360), (0, 200), (120, 250), (0, 30), (700, 1150), (550, 1000), (1, 365)]
    
    def multi_objective(x):
        return scalarized_fitness(
            x, st.session_state.costs, predictor,
            w_strength, w_carbon, w_cost, advanced=use_advanced_chemistry,
        )

    # ---- NSGA-II / NSGA-III: true multi-objective Pareto front --------------
    if is_nsga and st.button(f"Run {algorithm}"):
        seed = None
        if warm and warm_target:
            with st.spinner("Warm-starting from inverse design…"):
                seed = bayesian.sample_posterior(float(warm_target), n_samples=int(nsga_pop), method="auto")
        algo_key = "nsga3" if "III" in algorithm else "nsga2"
        with st.spinner(f"Running {algorithm} — mapping the trade-off surface…"):
            st.session_state.nsga_out = run_nsga(
                predictor, advanced=use_advanced_chemistry, costs=st.session_state.costs,
                algorithm=algo_key, pop_size=int(nsga_pop), n_gen=int(nsga_gen),
                seed_population=seed,
            )

    if is_nsga and st.session_state.get("nsga_out") is not None:
        nsga_out = st.session_state.nsga_out
        st.success(f"{nsga_out['algorithm']}: {nsga_out['front_size']} non-dominated mixes on the Pareto front.")

        h = nsga_out["history"]
        conv = go.Figure()
        conv.add_trace(go.Scatter(y=h["best_strength"], name="Best Strength (MPa)", line=dict(color="#00E676")))
        conv.add_trace(go.Scatter(y=h["min_carbon"], name="Min Carbon (kg/m³)", line=dict(color="#FFB300"), yaxis="y2"))
        conv.add_trace(go.Scatter(y=h["min_cost"], name="Min Cost ($/m³)", line=dict(color="#E91E63"), yaxis="y3"))
        conv.update_layout(
            template="plotly_dark", title="Per-objective best over generations", xaxis_title="Generation",
            yaxis=dict(title=dict(text="Strength", font=dict(color="#00E676")), tickfont=dict(color="#00E676")),
            yaxis2=dict(title=dict(text="Carbon", font=dict(color="#FFB300")), tickfont=dict(color="#FFB300"), overlaying="y", side="right"),
            yaxis3=dict(title=dict(text="Cost", font=dict(color="#E91E63")), tickfont=dict(color="#E91E63"), overlaying="y", side="right", anchor="free", autoshift=True),
            height=340, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.2))
        st.plotly_chart(conv, use_container_width=True)

        front_fig = go.Figure(go.Scatter3d(
            x=nsga_out["strength"], y=nsga_out["carbon"], z=nsga_out["cost"], mode="markers",
            marker=dict(size=5, color=nsga_out["strength"], colorscale="Viridis", opacity=0.9),
            text=[" · ".join(f"{p}:{v:.0f}" for p, v in zip(param_names, m)) for m in nsga_out["mixes"]],
            hoverinfo="text", name="Pareto front"))
        front_fig.update_layout(
            template="plotly_dark", height=600, margin=dict(l=0, r=0, t=10, b=0),
            scene=dict(xaxis_title="Strength (MPa)", yaxis_title="Carbon (kg/m³)", zaxis_title="Cost ($/m³)"))
        st.plotly_chart(front_fig, use_container_width=True)

        st.markdown("**Pick a mix from the front → Mix A:**")
        i_s = int(np.argmax(nsga_out["strength"]))
        i_c = int(np.argmin(nsga_out["carbon"]))
        i_m = int(np.argmin(nsga_out["cost"]))
        pk1, pk2, pk3 = st.columns(3)
        if pk1.button(f"Max strength · {nsga_out['strength'][i_s]:.0f} MPa"):
            st.session_state.mix_a = np.array(nsga_out["mixes"][i_s]); st.success("Loaded into Mix A.")
        if pk2.button(f"Min carbon · {nsga_out['carbon'][i_c]:.0f} kg CO₂"):
            st.session_state.mix_a = np.array(nsga_out["mixes"][i_c]); st.success("Loaded into Mix A.")
        if pk3.button(f"Min cost · ${nsga_out['cost'][i_m]:.0f}"):
            st.session_state.mix_a = np.array(nsga_out["mixes"][i_m]); st.success("Loaded into Mix A.")

        st.dataframe(
            pd.DataFrame({"Strength": nsga_out["strength"], "Carbon": nsga_out["carbon"], "Cost": nsga_out["cost"]}).round(1),
            use_container_width=True, height=240)

        st.subheader("Pareto front heatmap")
        fh = pd.DataFrame(nsga_out["mixes"], columns=param_names)
        hm = px.imshow(fh.T, labels=dict(x="Front mix", y="Parameter", color="Value"),
                       color_continuous_scale="Viridis", template="plotly_dark")
        hm.update_layout(height=420)
        st.plotly_chart(hm, use_container_width=True)

    if (not is_nsga) and st.button("Run Live Optimization"):
        progress_bar = st.progress(0)
        col_plots_1, col_plots_2 = st.columns(2)
        with col_plots_1:
            convergence_placeholder = st.empty()
        with col_plots_2:
            gene_dist_placeholder = st.empty()
        
        pareto_placeholder = st.empty()
        
        history_best, history_avg, history_diversity = [], [], []
        history_metrics = {"strength": [], "carbon": [], "cost": []}
        all_pareto_points = []
        
        if algorithm == "Genetic Algorithm (GA)":
            optimizer = GeneticOptimizer(multi_objective, bounds, pop_size=pop_size)
            total_steps = n_gens
            
            for g in range(n_gens):
                stats = optimizer.step()
                history_best.append(stats["best_fitness"])
                history_avg.append(stats["avg_fitness"])
                
                # Calculate population diversity (mean standard deviation across genes)
                diversity = np.mean(np.std(optimizer.population, axis=0))
                history_diversity.append(diversity)
                
                # Track metrics for the best individual
                best_ind, _ = optimizer.get_best()
                best_d = {k: v for k, v in zip(param_names, best_ind)}
                history_metrics["strength"].append(predictor.predict(best_ind))
                history_metrics["carbon"].append(carbon_for_mode(best_d, use_advanced_chemistry))
                history_metrics["cost"].append(calculate_mix_cost(best_d, st.session_state.costs))
                
                for ind in optimizer.population:
                    d = {k: v for k, v in zip(param_names, ind)}
                    all_pareto_points.append({
                        "Strength": predictor.predict(ind),
                        "Carbon": carbon_for_mode(d, use_advanced_chemistry),
                        "Cost": calculate_mix_cost(d, st.session_state.costs),
                        "Mix": "<br>".join([f"{param_names[i]}: {ind[i]:.1f}" for i in range(8)])
                    })
                
                progress_bar.progress((g + 1) / total_steps)
                
                # Convergence & Diversity Plot
                conv_fig = go.Figure()
                conv_fig.add_trace(go.Scatter(y=history_best, mode="lines+markers", name="Best Fitness", line=dict(color="#00E676", width=3)))
                conv_fig.add_trace(go.Scatter(y=history_avg, mode="lines", name="Avg Fitness", line=dict(color="#2979FF", dash="dash")))
                conv_fig.add_trace(go.Scatter(y=history_diversity, mode="lines", name="Gen Diversity", line=dict(color="#FFB300"), yaxis="y2"))
                conv_fig.update_layout(
                    template="plotly_dark", title="GA Performance & Genetic Diversity",
                    xaxis_title="Generation", yaxis_title="Fitness Score",
                    yaxis2=dict(title="Diversity (Std Dev)", overlaying="y", side="right"),
                    height=350, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.1)
                )
                convergence_placeholder.plotly_chart(conv_fig, use_container_width=True)
                
                # Gene Distribution Plot (Violin)
                pop_df = pd.DataFrame(optimizer.population, columns=[p.replace('_', ' ').title() for p in param_names])
                gene_fig = px.violin(pop_df.melt(), y="value", x="variable", color="variable", box=True, points=False, template="plotly_dark", title="Population Gene Pool Distribution")
                gene_fig.update_layout(height=350, showlegend=False, xaxis_title="", yaxis_title="Mass (kg/m³)", margin=dict(l=10, r=10, t=40, b=10))
                gene_dist_placeholder.plotly_chart(gene_fig, use_container_width=True)
                
                # 3D Pareto
                pareto_df = pd.DataFrame(all_pareto_points[-pop_size:])
                pareto_fig = px.scatter_3d(pareto_df, x="Strength", y="Carbon", z="Cost", color="Strength", hover_data=["Mix"], template="plotly_dark", title="Current population (evaluated mixes)")
                pareto_fig.update_layout(height=650, margin=dict(l=0, r=0, t=30, b=0))
                pareto_placeholder.plotly_chart(pareto_fig, use_container_width=True)
            
            final_pop = optimizer.population
        
        else:  # Simulated Annealing
            sa = SimulatedAnnealing(multi_objective, bounds, initial_temp=initial_temp, cooling_rate=cooling_rate, maximize=True)
            total_steps = n_steps
            
            for step in range(n_steps):
                stats = sa.step()
                history_best.append(stats["best_fitness"])
                history_avg.append(stats["current_fitness"])
                
                # Track metrics for the best solution
                best_sol = sa.best
                best_d = {k: v for k, v in zip(param_names, best_sol)}
                history_metrics["strength"].append(predictor.predict(best_sol))
                history_metrics["carbon"].append(carbon_for_mode(best_d, use_advanced_chemistry))
                history_metrics["cost"].append(calculate_mix_cost(best_d, st.session_state.costs))

                for sol in [sa.current, sa.best]:
                    d = {k: v for k, v in zip(param_names, sol)}
                    all_pareto_points.append({
                        "Strength": predictor.predict(sol),
                        "Carbon": carbon_for_mode(d, use_advanced_chemistry),
                        "Cost": calculate_mix_cost(d, st.session_state.costs),
                        "Mix": "<br>".join([f"{param_names[i]}: {sol[i]:.1f}" for i in range(8)])
                    })
                
                progress_bar.progress((step + 1) / total_steps)
                
                conv_fig = go.Figure()
                conv_fig.add_trace(go.Scatter(y=history_best, mode="lines+markers", name="Global Best", line=dict(color="#00E676")))
                conv_fig.add_trace(go.Scatter(y=history_avg, mode="lines", name="Current Temp Sol", line=dict(color="#FF5722", dash="dot")))
                conv_fig.update_layout(template="plotly_dark", title=f"SA Trace (T={stats['temperature']:.2f})", xaxis_title="Step", yaxis_title="Fitness", height=350)
                convergence_placeholder.plotly_chart(conv_fig, use_container_width=True)
                
                pareto_df = pd.DataFrame(all_pareto_points[-100:])
                pareto_fig = px.scatter_3d(pareto_df, x="Strength", y="Carbon", z="Cost", color="Strength", hover_data=["Mix"], template="plotly_dark", title="Annealing trajectory (evaluated mixes)")
                pareto_fig.update_layout(height=650)
                pareto_placeholder.plotly_chart(pareto_fig, use_container_width=True)
                
                if stats["temperature"] < 1e-6:
                    break
            
            final_pop = np.array([sa.best])
            best_ind = sa.best
        
        st.success(f"Optimization complete! Best Fitness: {history_best[-1]:.2f}")
        
        # Display Best Solution Breakdown
        b_col1, b_col2, b_col3, b_col4 = st.columns(4)
        best_d = {k: v for k, v in zip(param_names, best_ind)}
        with b_col1:
            st.metric("Best Strength", f"{history_metrics['strength'][-1]:.1f} MPa")
        with b_col2:
            st.metric("Best Carbon", f"{history_metrics['carbon'][-1]:.1f} kg/m³")
        with b_col3:
            st.metric("Best Cost", f"${history_metrics['cost'][-1]:.2f}/m³")
        with b_col4:
            st.metric("Best Fitness", f"{history_best[-1]:.2f}")

        # Metrics Evolution Plot
        metrics_fig = go.Figure()
        metrics_fig.add_trace(go.Scatter(y=history_metrics["strength"], name="Strength (MPa)", line=dict(color="#00E676")))
        metrics_fig.add_trace(go.Scatter(y=history_metrics["carbon"], name="Carbon (kg/m³)", line=dict(color="#FFB300"), yaxis="y2"))
        metrics_fig.add_trace(go.Scatter(y=history_metrics["cost"], name="Cost ($/m³)", line=dict(color="#E91E63"), yaxis="y3"))
        
        metrics_fig.update_layout(
            template="plotly_dark", title="Evolution of Best Solution Metrics",
            xaxis_title="Generation/Step",
            yaxis=dict(title=dict(text="Strength (MPa)", font=dict(color="#00E676")), tickfont=dict(color="#00E676")),
            yaxis2=dict(title=dict(text="Carbon (kg/m³)", font=dict(color="#FFB300")), tickfont=dict(color="#FFB300"), overlaying="y", side="right"),
            yaxis3=dict(title=dict(text="Cost ($/m³)", font=dict(color="#E91E63")), tickfont=dict(color="#E91E63"), overlaying="y", side="right", anchor="free", autoshift=True),
            height=400, showlegend=True, margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(metrics_fig, use_container_width=True)

        # True Pareto front: the non-dominated subset of every mix evaluated during
        # the search (max strength, min carbon, min cost). The scalarized search above
        # produces a cloud; this extracts the actual trade-off surface from it.
        st.subheader("Pareto-optimal set (non-dominated mixes)")
        all_df = pd.DataFrame(all_pareto_points)
        if len(all_df) > 3000:
            all_df = all_df.tail(3000).reset_index(drop=True)
        on_front = pareto_front_mask(all_df["Strength"], all_df["Carbon"], all_df["Cost"])
        front_df = all_df[on_front].sort_values("Strength")
        st.caption(
            f"{on_front.sum()} non-dominated mixes out of {len(all_df)} evaluated — "
            "each one cannot be improved on strength, carbon, or cost without worsening another."
        )
        front_fig = go.Figure()
        front_fig.add_trace(go.Scatter3d(
            x=all_df["Strength"], y=all_df["Carbon"], z=all_df["Cost"], mode="markers",
            marker=dict(size=2, color="#555", opacity=0.35), name="evaluated", hoverinfo="skip"))
        front_fig.add_trace(go.Scatter3d(
            x=front_df["Strength"], y=front_df["Carbon"], z=front_df["Cost"], mode="markers",
            marker=dict(size=5, color=front_df["Strength"], colorscale="Viridis", opacity=0.95),
            text=front_df["Mix"], hoverinfo="text", name="Pareto front"))
        front_fig.update_layout(
            template="plotly_dark", height=600, margin=dict(l=0, r=0, t=10, b=0),
            scene=dict(xaxis_title="Strength (MPa)", yaxis_title="Carbon (kg/m³)", zaxis_title="Cost ($/m³)"),
            legend=dict(orientation="h", y=1.05))
        st.plotly_chart(front_fig, use_container_width=True)

        st.subheader("Population Heatmap & Genetic Signatures")
        pop_df = pd.DataFrame(final_pop, columns=param_names)
        heatmap_fig = px.imshow(pop_df.T, labels=dict(x="Individual", y="Parameter", color="Value"), color_continuous_scale="Viridis", template="plotly_dark")
        heatmap_fig.update_layout(height=450)
        st.plotly_chart(heatmap_fig, use_container_width=True)
    
    st.markdown("""
    <div class="footnote">
    <strong>Philosophical Note on Stochastic Search:</strong><br>
    The optimization problem in concrete formulation is non-convex and high-dimensional. We employ metaheuristics 
    because they do not rely on local gradients, which are often noisy or unavailable in empirical models.<br><br>
    
    <em>Genetic Algorithm (GA) Mechanics:</em> GA simulates phenotypic evolution. The 'fitness' is a mathematical 
    representation of structural requirement (strength) penalized by environmental and economic constraints. 
    The <strong>Gen Diversity</strong> metric tracks the standard deviation of the population's genes. 
    A rapid collapse in diversity suggests 'Premature Convergence'—where the population settles into a 
    sub-optimal local peak. The <strong>Violin Plots</strong> visualize this in real-time: watch the 
    distribution 'blobs' shrink as the population converges on a specific molecular recipe.<br><br>
    
    <em>Simulated Annealing (SA) Mechanics:</em> Where GA is a population-level search, SA is a single-agent 
    trajectory. It models the thermodynamic probability of a system changing states. At high 
    temperatures (initial steps), the walker is allowed to accept 'worse' solutions to escape local 
    minimums. As the 'Temperature' (T) cools, the walker becomes increasingly local, refining 
    its current position into a globally optimal configuration.<br><br>
    
    <em>Pareto Frontier:</em> In multi-objective design, 'the best mix' does not exist. Instead there
    is a set of points where one objective (e.g., Cost) cannot be improved without sacrificing another
    (e.g., Strength) — the Pareto front. The weighted objective (Strength - w<sub>C</sub>·Carbon -
    w<sub>$</sub>·Cost) drives a single search direction, so the live scatter above is the cloud of
    <em>evaluated</em> mixes, not the front itself. The "Pareto-optimal set" plot extracts the true
    <strong>non-dominated</strong> subset of every evaluated mix, so the label reflects what is
    actually computed. (A dedicated multi-objective method such as NSGA-II would populate the whole
    front more evenly; that is future work.)
    </div>
    """, unsafe_allow_html=True)

with tab4:
    st.header("Calibration")
    st.markdown("""
    **How to use:** Upload a CSV of actual lab test results to improve the model's accuracy. 
    Required columns: `cement, slag, ash, water, superplasticizer, coarse_agg, fine_agg, age, strength`.
    """)
    
    uploaded_data = st.file_uploader("Upload Lab Results (CSV)", type="csv")
    if uploaded_data:
        new_df = pd.read_csv(uploaded_data)
        st.write("Preview of incoming data:", new_df.head())
        error = validate_lab_csv(new_df)
        if error:
            st.error(f"Cannot merge this file: {error}")
        elif st.button("Merge & Retrain Digital Twin"):
            with st.spinner("Incorporating new field data..."):
                append_experimental_results(new_df)
                predictor.train()
                load_resources.clear()  # invalidate the cached predictor
                st.success("Model calibrated with actual field results!")
                st.balloons()
    
    st.markdown("""
    <div class="footnote">
    <strong>About Calibration:</strong> Machine learning models are only as good as their training data. 
    The UCI dataset was collected in Taiwan in the 1990s and may not reflect modern admixtures, regional 
    materials, or your specific cement suppliers. The calibration feature allows you to upload actual 
    laboratory test results (from destructive cylinder breaks or NDE methods like rebound hammer or 
    ultrasonic pulse velocity). These are appended to a local "overlay" dataset and the XGBoost model is 
    retrained to incorporate your field experience. Over time, this creates a Digital Twin that reflects 
    your specific materials and processes.
    </div>
    """, unsafe_allow_html=True)

with tab_workflow:
    with open("docs/WORKFLOW.md", "r", encoding="utf-8") as f:
        workflow_md = f.read()
    # Streamlit's markdown doesn't render mermaid; drop the diagram block (it renders
    # on GitHub) and keep the ordered step-by-step text.
    workflow_md = re.sub(r"```mermaid.*?```",
                         "_(flow diagram renders on GitHub — the ordered steps are below)_",
                         workflow_md, flags=re.DOTALL)
    st.markdown(workflow_md)

with tab5:
    st.header("Technical Report: Generative Mix Design")
    with open("docs/TECHNICAL_REPORT.md", "r", encoding="utf-8") as f:
        report_content = f.read()
    st.markdown(report_content)

with tab6:
    st.header("References & Resources")
    
    st.subheader("Software & Libraries")
    st.markdown("""
    | Component | Library | Version | Link |
    |-----------|---------|---------|------|
    | ML Prediction | XGBoost | 3.x | [xgboost.readthedocs.io](https://xgboost.readthedocs.io/) |
    | Bayesian Inference | BayesFlow | 1.x | [github.com/stefanradev93/BayesFlow](https://github.com/stefanradev93/BayesFlow) |
    | Deep Learning | TensorFlow | 2.14.* | [tensorflow.org](https://www.tensorflow.org/) |
    | Probabilistic | TensorFlow Probability | 0.22.* | [tensorflow.org/probability](https://www.tensorflow.org/probability) |
    | Visualization | Plotly | 5.18+ | [plotly.com/python](https://plotly.com/python/) |
    | Web Framework | Streamlit | 1.30+ | [streamlit.io](https://streamlit.io/) |
    | Scientific | NumPy, SciPy, Pandas | — | [numpy.org](https://numpy.org/) |
    """)
    
    st.subheader("Primary Research References")
    st.markdown("""
    **Machine Learning & Concrete Prediction**
    
    1. **Yeh, I-C. (1998).** "Modeling of strength of high-performance concrete using artificial neural networks." 
       *Cement and Concrete Research*, 28(12), 1797-1808. 
       *[The foundational dataset for this project. Established that nonlinear models outperform traditional regression for concrete.]*
    
    2. **Chen, T. & Guestrin, C. (2016).** "XGBoost: A Scalable Tree Boosting System." 
       *Proceedings of the 22nd ACM SIGKDD*, 785-794.
       *[The algorithm behind our forward predictor. Gradient boosting with regularization.]*
    
    3. **DeRousseau, M.A. et al. (2019).** "A comparison of machine learning methods for predicting compressive strength of concrete." 
       *Construction and Building Materials*, 161, 164-176.
       *[Benchmark study comparing RF, SVR, ANN, and XGBoost on concrete datasets.]*
    
    **Bayesian & Amortized Inference**
    
    4. **Radev, S.T. et al. (2020).** "BayesFlow: Learning complex stochastic models with invertible neural networks." 
       *IEEE Transactions on Neural Networks and Learning Systems*, 31(11), 5051-5064.
       *[The theoretical foundation for our inverse design engine.]*
    
    5. **Papamakarios, G. et al. (2021).** "Normalizing Flows for Probabilistic Modeling and Inference." 
       *Journal of Machine Learning Research*, 22(57), 1-64.
       *[Comprehensive review of the normalizing flow architecture we use for posterior estimation.]*
    
    6. **Cranmer, K. et al. (2020).** "The frontier of simulation-based inference." 
       *PNAS*, 117(48), 30055-30062.
       *[Contextualizes amortized inference within broader scientific simulation.]*
    
    **Cement Chemistry**
    
    7. **Bogue, R.H. (1929).** "Calculation of the Compounds in Portland Cement." 
       *Industrial & Engineering Chemistry Analytical Edition*, 1(4), 192-197.
       *[The classic calculation for estimating clinker phases from oxide composition.]*
    
    8. **Taylor, H.F.W. (1997).** *Cement Chemistry*, 2nd Edition. Thomas Telford.
       *[The definitive textbook on cement hydration—our Tier 2 chemistry model is inspired by chapters 6-8.]*
    
    9. **Lothenbach, B., Scrivener, K., & Hooton, R.D. (2011).** "Supplementary cementitious materials." 
       *Cement and Concrete Research*, 41(12), 1244-1256.
       *[Comprehensive review of SCM chemistry that informs our pozzolanic reaction model.]*
    
    10. **Scrivener, K.L. et al. (2015).** "TC 238-SCM: Hydration and microstructure of concrete with SCMs." 
        *Materials and Structures*, 48, 835-862.
        *[State-of-the-art on SCM hydration mechanisms.]*
    
    **Sustainability & Carbon Accounting**
    
    11. **WBCSD/CSI (2013).** "The Cement CO2 and Energy Protocol: CO2 and Energy Accounting 
        and Reporting Standard for the Cement Industry." World Business Council for Sustainable Development.
        *[Industry-standard methodology for carbon accounting that our chemistry layer follows.]*
    
    12. **Habert, G. et al. (2020).** "Environmental impacts and decarbonization strategies in the cement industry." 
        *Nature Reviews Earth & Environment*, 1, 559-573.
        *[Modern perspective on cement decarbonization—motivates our multi-objective optimization.]*
    
    **Optimization & Metaheuristics**
    
    13. **Holland, J.H. (1975).** *Adaptation in Natural and Artificial Systems*. University of Michigan Press.
        *[The original text on genetic algorithms.]*
    
    14. **Kirkpatrick, S., Gelatt, C.D., & Vecchi, M.P. (1983).** "Optimization by Simulated Annealing." 
        *Science*, 220(4598), 671-680.
        *[The foundational paper on simulated annealing—our SA implementation follows this formulation.]*
    
    15. **Deb, K. (2001).** *Multi-Objective Optimization Using Evolutionary Algorithms*. Wiley.
        *[Theoretical framework for Pareto optimization; explains scalarization vs. Pareto dominance.]*
    """)
    
    st.subheader("Further Reading")
    st.markdown("""
    - **Neville, A.M. (2011).** *Properties of Concrete*, 5th Edition. Pearson. 
      [Comprehensive reference on concrete materials and behavior]
    
    - **Mehta, P.K. & Monteiro, P.J.M. (2014).** *Concrete: Microstructure, Properties, and Materials*, 4th Edition. McGraw-Hill.
      [Authoritative textbook covering concrete from microstructure to durability]
    
    - **ACI 211.1-91.** "Standard Practice for Selecting Proportions for Normal, Heavyweight, and Mass Concrete."
      [Traditional mix design methodology]
    """)
    
    st.subheader("Acknowledgments")
    st.markdown("""
    This tool was developed to accelerate sustainable concrete design and democratize access to advanced 
    optimization techniques. We acknowledge the researchers who made their datasets publicly available, 
    the open-source community (NumPy, SciPy, TensorFlow, Streamlit, Plotly), and the cement science 
    community whose decades of research make computational models possible.
    
    Special thanks to the UCI Machine Learning Repository for hosting the Concrete Compressive Strength dataset.
    """)
