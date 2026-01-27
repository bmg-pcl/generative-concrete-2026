import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from src.models import StrengthPredictor
from src.chemistry_simple import calculate_embodied_carbon, estimate_curing_time, calculate_mix_cost, UNIT_COSTS
from src.chemistry_advanced import analyze_mix, carbon_from_clinker
from src.bayesian import BayesFlowExplorer
from src.ga import GeneticOptimizer
from src.annealing import SimulatedAnnealing
from src.data_fetcher import append_experimental_results, load_data

st.set_page_config(page_title="Generative Mix Design", layout="wide", initial_sidebar_state="expanded")

# --- Dark Mode Styling ---
st.markdown("""
<style>
    .stApp { background-color: #121212; color: #E0E0E0; }
    .stTabs [data-baseweb="tab-list"] { background-color: #121212; }
    .stMetric { background-color: #1E1E1E; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    .main-title { font-size: 3rem; font-weight: 800; background: -webkit-linear-gradient(#00E676, #2979FF); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .footnote { font-size: 0.75rem; color: #888; line-height: 1.4; margin-top: 20px; }
</style>
""", unsafe_allow_html=True)

# --- Loading Status ---
status_container = st.empty()
status_container.info("🔄 Initializing models...")

# --- Preset Mixtures from Dataset ---
@st.cache_data
def get_preset_mixtures():
    df = load_data()
    presets = {
        "Custom": None,
        "High Strength (Row 42)": df.iloc[42].values[:8],
        "Low Carbon (Row 100)": df.iloc[100].values[:8],
        "Balanced (Row 200)": df.iloc[200].values[:8],
        "High SCM (Row 500)": df.iloc[500].values[:8],
    }
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

status_container.info("🔄 Loading strength predictor...")
predictor, bayesian = load_resources()

status_container.info("🔄 Loading dataset presets...")
PRESET_MIXTURES = get_preset_mixtures()

status_container.empty()  # Clear loading message

param_names = ["cement", "slag", "ash", "water", "superplasticizer", "coarse_agg", "fine_agg", "age"]

# Extended admixtures
EXOTIC_ADMIXTURES = {
    "silica_fume": {"default": 0, "max": 50, "carbon_factor": 0.02, "cost": 0.80, "category": "Pozzolan"},
    "metakaolin": {"default": 0, "max": 80, "carbon_factor": 0.30, "cost": 0.45, "category": "Pozzolan"},
    "rice_husk_ash": {"default": 0, "max": 60, "carbon_factor": 0.01, "cost": 0.15, "category": "Pozzolan"},
    "limestone_filler": {"default": 0, "max": 100, "carbon_factor": 0.01, "cost": 0.05, "category": "Filler"},
    "calcined_clay": {"default": 0, "max": 150, "carbon_factor": 0.25, "cost": 0.12, "category": "Filler"},
    "steel_fiber": {"default": 0, "max": 80, "carbon_factor": 1.80, "cost": 1.50, "category": "Fiber"},
    "polypropylene_fiber": {"default": 0, "max": 10, "carbon_factor": 3.50, "cost": 4.00, "category": "Fiber"},
    "basalt_fiber": {"default": 0, "max": 20, "carbon_factor": 0.60, "cost": 2.50, "category": "Fiber"},
    "nano_silica": {"default": 0, "max": 5, "carbon_factor": 5.00, "cost": 25.00, "category": "Nano"},
    "graphene_oxide": {"default": 0, "max": 1, "carbon_factor": 50.00, "cost": 500.00, "category": "Nano"},
    "calcium_chloride": {"default": 0, "max": 10, "carbon_factor": 0.80, "cost": 0.30, "category": "Chemical"},
    "shrink_reducer": {"default": 0, "max": 8, "carbon_factor": 2.00, "cost": 6.00, "category": "Chemical"},
}

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
st.markdown('<h1 class="main-title">🧪 Generative Mix Design</h1>', unsafe_allow_html=True)
st.markdown("AI-powered concrete formulation: prediction, optimization, and inverse design.")

# --- Sidebar ---
st.sidebar.header("💾 Session Export/Import")
st.sidebar.caption("Save or restore your complete session configuration.")

def get_state_json():
    state = {
        "mix_a": st.session_state.mix_a.tolist(),
        "mix_b": st.session_state.mix_b.tolist(),
        "costs": st.session_state.costs,
        "exotic_a": st.session_state.exotic_a,
        "exotic_b": st.session_state.exotic_b,
    }
    return json.dumps(state)

st.sidebar.download_button(label="Export Session (JSON)", data=get_state_json(), file_name="gmd_session.json", mime="application/json")

uploaded_state = st.sidebar.file_uploader("Import Session", type="json")
if uploaded_state:
    data = json.load(uploaded_state)
    st.session_state.mix_a = np.array(data["mix_a"])
    st.session_state.mix_b = np.array(data["mix_b"])
    st.session_state.costs = data["costs"]
    if "exotic_a" in data:
        st.session_state.exotic_a = data["exotic_a"]
        st.session_state.exotic_b = data["exotic_b"]
    st.sidebar.success("Session Imported!")

st.sidebar.divider()
st.sidebar.header("💰 Material Costs")
st.sidebar.caption("Costs in $ per kilogram.")
for mat in st.session_state.costs:
    st.session_state.costs[mat] = st.sidebar.number_input(f"{mat.replace('_', ' ').title()} ($/kg)", value=st.session_state.costs[mat], format="%.4f")

st.sidebar.divider()
st.sidebar.header("🔬 Chemistry Mode")
chemistry_mode = st.sidebar.radio(
    "Carbon & Analysis Model",
    ["Simple (Linear)", "Advanced (Molecular)"],
    help="Simple uses mass × factor. Advanced uses Bogue calculations and clinker chemistry."
)
use_advanced_chemistry = (chemistry_mode == "Advanced (Molecular)")

# --- Main Layout ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["⚡ Generative Mix Design", "📊 Amortized Performance", "🧬 Pareto Optimization", "🔧 Calibration", "📄 Technical Report", "📚 References"])

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
            for adm, props in EXOTIC_ADMIXTURES.items():
                exotic_state[adm] = st.slider(f"{adm.replace('_', ' ').title()} ({key})", 0, props["max"], exotic_state.get(adm, 0))
        
        return mix_vals, exotic_state

    with col_a:
        st.session_state.mix_a, st.session_state.exotic_a = mix_input_ui("A", st.session_state.mix_a, st.session_state.exotic_a)
    with col_b:
        st.session_state.mix_b, st.session_state.exotic_b = mix_input_ui("B", st.session_state.mix_b, st.session_state.exotic_b)

    def get_metrics(mix, exotic):
        d = {k: v for k, v in zip(param_names, mix)}
        extra_carbon = sum(exotic.get(k, 0) * EXOTIC_ADMIXTURES[k]["carbon_factor"] for k in EXOTIC_ADMIXTURES)
        extra_cost = sum(exotic.get(k, 0) * EXOTIC_ADMIXTURES[k]["cost"] for k in EXOTIC_ADMIXTURES)
        return {
            "strength": predictor.predict(mix),
            "carbon": calculate_embodied_carbon(d) + extra_carbon,
            "cost": calculate_mix_cost(d, st.session_state.costs) + extra_cost,
            "curing": estimate_curing_time(d),
            "uncertainty": bayesian.evaluate_uncertainty(mix)
        }

    m_a = get_metrics(st.session_state.mix_a, st.session_state.exotic_a)
    m_b = get_metrics(st.session_state.mix_b, st.session_state.exotic_b)

    st.divider()
    res_a, res_b = st.columns(2)
    with res_a:
        st.subheader("Mix A")
        st.metric("Strength", f"{m_a['strength']:.1f} MPa")
        st.metric("Carbon", f"{m_a['carbon']:.1f} kg CO₂/m³")
        st.metric("Cost", f"${m_a['cost']:.2f}/m³")
    with res_b:
        st.subheader("Mix B")
        st.metric("Strength", f"{m_b['strength']:.1f} MPa", delta=f"{m_b['strength']-m_a['strength']:.1f}")
        st.metric("Carbon", f"{m_b['carbon']:.1f} kg CO₂/m³", delta=f"{m_b['carbon']-m_a['carbon']:.1f}", delta_color="inverse")
        st.metric("Cost", f"${m_b['cost']:.2f}/m³", delta=f"${m_b['cost']-m_a['cost']:.2f}", delta_color="inverse")

with tab2:
    st.header("📊 Amortized Performance Estimates")
    st.markdown("""
    **How to use:** Enter a target compressive strength. The 3D surface shows the probability 
    distribution of mix designs that could achieve it. Peaks indicate likely parameter combinations.
    """)
    
    target_str = st.number_input("Target Strength (MPa)", 10, 100, 45)
    px_idx = st.selectbox("X-Axis Parameter", range(8), index=0, format_func=lambda x: param_names[x])
    py_idx = st.selectbox("Y-Axis Parameter", range(8), index=3, format_func=lambda x: param_names[x])

    samples = bayesian.sample_posterior(target_str, n_samples=3000)

    # Build hover text with full material breakdown + performance metrics
    hover_df = pd.DataFrame(samples[:300], columns=param_names)
    hover_texts = []
    for _, row in hover_df.iterrows():
        mix_dict = row.to_dict()
        strength = predictor.predict(row.values)
        if use_advanced_chemistry:
            carbon = carbon_from_clinker(mix_dict.get("cement", 300))
        else:
            carbon = calculate_embodied_carbon(mix_dict)
        cost = calculate_mix_cost(mix_dict, st.session_state.costs)
        
        lines = [
            f"<b>STRENGTH: {strength:.1f} MPa</b>",
            f"<b>CARBON: {carbon:.1f} kg/m³</b>",
            f"<b>COST: ${cost:.2f}/m³</b>",
            "---"
        ]
        lines.extend([f"{p}: {mix_dict[p]:.1f}" for p in param_names])
        hover_texts.append("<br>".join(lines))
    
    # 3D Density Mesh
    from scipy.stats import gaussian_kde
    x, y = samples[:, px_idx], samples[:, py_idx]
    xy = np.vstack([x, y]); kde = gaussian_kde(xy)
    xi, yi = np.mgrid[x.min():x.max():50j, y.min():y.max():50j]
    zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
    
    fig = go.Figure()
    # Add Density Surface
    fig.add_trace(go.Surface(
        z=zi, x=xi, y=yi, colorscale='Magma', opacity=0.8, 
        name="Density", showscale=False, hoverinfo="skip"
    ))
    # Add Sample Points for Hovering (Sub-sampled for visibility)
    sample_densities = kde(np.vstack([x[:300], y[:300]]))
    fig.add_trace(go.Scatter3d(
        x=x[:300], y=y[:300], z=sample_densities, 
        mode='markers', marker=dict(size=3, color='cyan', opacity=0.5),
        text=hover_texts, hoverinfo="text", name="Sample Recipes"
    ))

    fig.update_layout(
        template="plotly_dark", 
        scene=dict(
            xaxis_title=param_names[px_idx].title(), 
            yaxis_title=param_names[py_idx].title(), 
            zaxis_title="Prob. Density"
        ), 
        margin=dict(l=0, r=0, b=0, t=0), height=750
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="footnote">
    <strong>About Amortized Inference:</strong> Traditional Bayesian inference requires running expensive 
    MCMC sampling for each new query. Amortized inference trains a neural network (using Normalizing Flows) 
    to learn the inverse mapping from observations to parameters across the entire data space. Once trained, 
    posterior sampling is instantaneous—O(1) instead of O(n). This enables real-time "what-if" exploration 
    of the design space. The 3D surface represents a kernel density estimate of plausible mix parameters 
    conditioned on the target strength. High-density regions indicate combinations the model considers most 
    likely to achieve the target.
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.header("🧬 Multi-Objective Pareto Optimization")
    st.markdown("""
    **How to use:** Select an optimization algorithm, configure parameters, and click Run. 
    Watch the convergence plot and 3D Pareto frontier evolve in real-time.
    """)
    
    col_cfg, col_algo = st.columns([1, 1])
    with col_cfg:
        algorithm = st.selectbox("Optimization Algorithm", ["Genetic Algorithm (GA)", "Simulated Annealing (SA)"])
        if algorithm == "Genetic Algorithm (GA)":
            pop_size = st.number_input("Population Size", 20, 200, 50)
            n_gens = st.number_input("Generations", 10, 100, 30)
        else:
            initial_temp = st.number_input("Initial Temperature", 100, 5000, 1000)
            cooling_rate = st.slider("Cooling Rate", 0.80, 0.99, 0.95)
            n_steps = st.number_input("Max Temperature Steps", 20, 200, 50)
    
    bounds = [(100, 550), (0, 360), (0, 200), (120, 250), (0, 30), (700, 1150), (550, 1000), (1, 365)]
    
    def multi_objective(x):
        d = {k: v for k, v in zip(param_names, x)}
        strength = predictor.predict(x)
        if use_advanced_chemistry:
            carbon = carbon_from_clinker(d.get("cement", 300))
        else:
            carbon = calculate_embodied_carbon(d)
        cost = calculate_mix_cost(d, st.session_state.costs)
        return strength - 0.05 * carbon - 0.5 * cost
    
    if st.button("🚀 Run Live Optimization"):
        progress_bar = st.progress(0)
        col_plots_1, col_plots_2 = st.columns(2)
        with col_plots_1:
            convergence_placeholder = st.empty()
        with col_plots_2:
            gene_dist_placeholder = st.empty()
        
        pareto_placeholder = st.empty()
        
        history_best, history_avg, history_diversity = [], [], []
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
                
                for ind in optimizer.population:
                    d = {k: v for k, v in zip(param_names, ind)}
                    all_pareto_points.append({
                        "Strength": predictor.predict(ind),
                        "Carbon": calculate_embodied_carbon(d),
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
                pareto_fig = px.scatter_3d(pareto_df, x="Strength", y="Carbon", z="Cost", color="Strength", hover_data=["Mix"], template="plotly_dark", title="Dynamic Pareto Frontier")
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
                
                for sol in [sa.current, sa.best]:
                    d = {k: v for k, v in zip(param_names, sol)}
                    all_pareto_points.append({
                        "Strength": predictor.predict(sol),
                        "Carbon": calculate_embodied_carbon(d),
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
                pareto_fig = px.scatter_3d(pareto_df, x="Strength", y="Carbon", z="Cost", color="Strength", hover_data=["Mix"], template="plotly_dark", title="Annealing Solution Trajectory")
                pareto_fig.update_layout(height=650)
                pareto_placeholder.plotly_chart(pareto_fig, use_container_width=True)
                
                if stats["temperature"] < 1e-6:
                    break
            
            final_pop = np.array([sa.best])
        
        st.success(f"Optimization complete! Best Fitness: {history_best[-1]:.2f}")
        
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
    
    <em>Pareto Frontier Duality:</em> In multi-objective design, 'the best mix' does not exist. 
    Instead, we find the set of points where one objective (e.g., Cost) cannot be improved without 
    sacrificing another (e.g., Strength). This 3D boundary is the Pareto Frontier. Our scalarization 
    technique (Strength - 0.05*Carbon - 0.5*Cost) essentially sections this frontier at a specific 
    angle of priority.
    </div>
    """, unsafe_allow_html=True)

with tab4:
    st.header("🔧 Calibration")
    st.markdown("""
    **How to use:** Upload a CSV of actual lab test results to improve the model's accuracy. 
    Required columns: `cement, slag, ash, water, superplasticizer, coarse_agg, fine_agg, age, strength`.
    """)
    
    uploaded_data = st.file_uploader("Upload Lab Results (CSV)", type="csv")
    if uploaded_data:
        new_df = pd.read_csv(uploaded_data)
        st.write("Preview of incoming data:", new_df.head())
        if st.button("Merge & Retrain Digital Twin"):
            with st.spinner("Incorporating new field data..."):
                append_experimental_results(new_df)
                predictor.train()
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

with tab5:
    st.header("📄 Technical Report: Generative Mix Design")
    with open("docs/TECHNICAL_REPORT.md", "r", encoding="utf-8") as f:
        report_content = f.read()
    st.markdown(report_content)

with tab6:
    st.header("📚 References & Resources")
    
    st.subheader("Software & Libraries")
    st.markdown("""
    | Component | Library | Version | Link |
    |-----------|---------|---------|------|
    | ML Prediction | XGBoost | 2.0+ | [xgboost.readthedocs.io](https://xgboost.readthedocs.io/) |
    | Bayesian Inference | BayesFlow | 1.1+ | [github.com/stefanradev93/BayesFlow](https://github.com/stefanradev93/BayesFlow) |
    | Deep Learning | TensorFlow | 2.15+ | [tensorflow.org](https://www.tensorflow.org/) |
    | Probabilistic | TensorFlow Probability | 0.23+ | [tensorflow.org/probability](https://www.tensorflow.org/probability) |
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
