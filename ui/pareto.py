"""Pareto Optimization tab — weighted GA/SA (live) and NSGA-II/III (true front)."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.ga import GeneticOptimizer
from src.annealing import SimulatedAnnealing
from src.nsga import run_nsga, pymoo_available
from src.chemistry_simple import calculate_mix_cost
from src.ui_logic import PARAM_NAMES, scalarized_fitness, carbon_for_mode, pareto_front_mask
from ui.context import AppContext
from ui.state import load_mix_into, EXPOSURE_NONE

param_names = list(PARAM_NAMES)

# R8.6 WP-U3: shared verbatim substring the pre-run placeholder always carries,
# so the results column is never blank -- asserted by tests/test_ux_u3.py.
RESULTS_PLACEHOLDER_TEXT = "Charts stream here generation-by-generation once you run"


def render_pareto(ctx: AppContext):
    predictor = ctx.predictor
    bayesian = ctx.bayesian
    use_advanced_chemistry = ctx.use_advanced_chemistry
    carbon_kwargs = ctx.carbon_kwargs
    robust_mode = ctx.robust
    design_age = ctx.design_age

    st.header("Multi-Objective Pareto Optimization")
    st.markdown("""
    **How to use:** Pick an algorithm and click Run. **GA / SA** search a *weighted* objective
    (you set the weights) and stream live; **NSGA-II / NSGA-III** are true multi-objective methods
    that map the whole **strength / carbon / cost** Pareto front in one run — no weights, and they
    can warm-start from the inverse-design flow. See the Workflow tab for when to use which.
    """)

    col_cfg, col_results = st.columns([1, 2])

    with col_cfg:
        algo_options = ["Genetic Algorithm (GA)", "Simulated Annealing (SA)"]
        if pymoo_available():
            algo_options += ["NSGA-II (multi-objective)", "NSGA-III (multi-objective)"]
        algorithm = st.selectbox("Optimization Algorithm", algo_options)
        is_nsga = "NSGA" in algorithm

        # R8.5 P4 compliance state, defined on every path (only meaningfully set
        # on the NSGA branch below -- GA/SA have no compliance-aware objective).
        require_compliance = False
        allow_unknown = False
        exposure_pack_id = exposure_class_id = EXPOSURE_NONE
        exposure_selected = False

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

            # --- R8.5 P4: compliance constraint (NSGA already has native
            # support -- run_nsga(..., compliance=(pack_id, class_id)) -- unlike
            # the inverse tab, no design_compliant() workaround needed here).
            exposure_pack_id = st.session_state.get("cfg_exposure_pack", EXPOSURE_NONE)
            exposure_class_id = st.session_state.get("cfg_exposure_class", EXPOSURE_NONE)
            exposure_selected = (exposure_pack_id != EXPOSURE_NONE
                                 and exposure_class_id != EXPOSURE_NONE)
            # Distinct keys from the Inverse Design tab's identically-purposed
            # toggle (cfg_require_compliance/cfg_compliance_allow_unknown):
            # Streamlit runs every tab's code on each script run (see
            # tests/test_app_smoke.py's docstring), so two tabs instantiating a
            # widget under the SAME key in the same run raises a duplicate-ID
            # error -- these need their own state, and a user reasonably wants
            # compliance required in one tab but not the other anyway.
            require_compliance = st.checkbox(
                "Require compliance (selected exposure class)",
                key="cfg_require_compliance_pareto", disabled=not exposure_selected,
                help=(
                    f"ON: front members are constrained toward "
                    f"{exposure_pack_id}.{exposure_class_id} during the search, "
                    "then every member is independently RE-VERIFIED with the "
                    "real check_compliance() engine — never inferred from the "
                    "search constraint alone."
                    if exposure_selected else
                    "Select an exposure pack AND class on the Config tab to "
                    "enable this."
                ),
            )
            allow_unknown = False
            if require_compliance and exposure_selected:
                allow_unknown = st.checkbox(
                    "Allow UNKNOWN rules", key="cfg_compliance_allow_unknown_pareto",
                    help="OFF (strict, default): a front member must verify PASS "
                         "on every rule. ON: a member whose evaluable rules all "
                         "PASS also counts, even if some rules are UNKNOWN "
                         "(unsourced by the pack).",
                )
            if require_compliance and exposure_selected and not allow_unknown:
                st.info(
                    f"**Strict compliance mode.** {exposure_pack_id}.{exposure_class_id} "
                    "omits one or more deemed-to-satisfy rules this model cannot "
                    "source (e.g. the SCM cap) — those are UNKNOWN, and strict "
                    "mode counts UNKNOWN as a violation. A front with zero "
                    "verified-PASS members may reflect that ceiling, not a "
                    "broken search — check \"Allow UNKNOWN rules\" to see "
                    "members passing every rule this pack actually sources."
                )

        if is_nsga:
            st.caption("NSGA maps the whole strength / carbon / cost trade-off surface — no scalar weights needed.")
        else:
            st.subheader("Objective Weights")
            st.caption(
                "These weights multiply the optimizer's raw predicted quantities directly — "
                "Strength in MPa, Carbon in kg CO₂/m³, Cost in $/m³ — not normalized "
                "percentages. The combined score is "
                "`w_strength·strength − w_carbon·carbon − w_cost·cost`, and because carbon "
                "and cost are numerically much larger than strength, their default weights "
                "(0.05 and 0.5) sit far below Strength's (1.0) just to keep the three terms "
                "roughly balanced in that subtraction — the slider ranges are not directly "
                "comparable across objectives."
            )
            w_strength = st.slider("Strength Weight", 0.0, 2.0, 1.0, help="Weight for maximizing compressive strength.")
            w_carbon = st.slider("Carbon Penalty Weight", 0.0, 1.0, 0.05, help="Weight for minimizing embodied carbon.")
            w_cost = st.slider("Cost Penalty Weight", 0.0, 1.0, 0.5, help="Weight for minimizing material cost.")

        st.divider()
        # R8.6 WP-U3: run controls + live feedback anchored here at the controls
        # column, so the generation counter / current-best readout sits right
        # next to the button and progress bar instead of being discoverable
        # only by scrolling into the results column below.
        progress_bar = None
        live_status = None
        if is_nsga:
            run_clicked = st.button(f"Run {algorithm}", use_container_width=True)
        else:
            run_clicked = st.button("Run Live Optimization", use_container_width=True)
            if run_clicked:
                progress_bar = st.progress(0.0)
                live_status = st.empty()

    bounds = [(100, 550), (0, 360), (0, 200), (120, 250), (0, 30), (700, 1150), (550, 1000), (1, 365)]

    def multi_objective(x):
        return scalarized_fitness(
            x, st.session_state.costs, predictor,
            w_strength, w_carbon, w_cost, advanced=use_advanced_chemistry,
            carbon_kwargs=carbon_kwargs, robust=robust_mode,
        )

    with col_results:
        # ---- NSGA-II / NSGA-III: true multi-objective Pareto front ----------
        if is_nsga:
            if run_clicked:
                seed = None
                if warm and warm_target:
                    with st.spinner("Warm-starting from inverse design…"):
                        seed = bayesian.sample_posterior(float(warm_target), n_samples=int(nsga_pop),
                                                         method="auto", age=design_age)
                algo_key = "nsga3" if "III" in algorithm else "nsga2"
                compliance_arg = (exposure_pack_id, exposure_class_id) \
                    if (require_compliance and exposure_selected) else None
                with st.spinner(f"Running {algorithm} — mapping the trade-off surface…"):
                    st.session_state.nsga_out = run_nsga(
                        predictor, advanced=use_advanced_chemistry, costs=st.session_state.costs,
                        algorithm=algo_key, pop_size=int(nsga_pop), n_gen=int(nsga_gen),
                        seed_population=seed, carbon_kwargs=carbon_kwargs, robust=robust_mode,
                        age=design_age, compliance=compliance_arg,
                    )
                    # R8.5 P4: the strict/allow-unknown choice, disclosed alongside the
                    # existing "compliance" block in the same output dict (run_nsga
                    # itself has no such concept -- its own "all_pass" is always strict;
                    # we apply the allow-unknown relaxation ourselves below).
                    st.session_state.nsga_out["compliance_mode"] = \
                        "allow_unknown" if allow_unknown else "strict"

            if st.session_state.get("nsga_out") is not None:
                nsga_out = st.session_state.nsga_out
                st.success(f"{nsga_out['algorithm']}: {nsga_out['front_size']} non-dominated mixes on the Pareto front.")

                # R8.5 P4: honest compliance disclosure. run_nsga's own "all_pass" is
                # always the STRICT reading (every rule PASS); when allow-unknown is
                # selected we count our own acceptance criterion (verdict != "FAIL") over
                # the same per-member `results`, never a laxer verdict than
                # check_compliance itself computed -- see ui/inverse.py's `_pick_compliant`
                # for the identical rule, applied here to a whole front instead of one pick.
                c = nsga_out.get("compliance")
                if c:
                    mode = nsga_out.get("compliance_mode", "strict")
                    if mode == "allow_unknown":
                        accepted = sum(1 for r in c["results"] if r["verdict"] != "FAIL")
                    else:
                        accepted = sum(1 for r in c["results"] if r["verdict"] == "PASS")
                    label = (f"Compliance ({mode}): {accepted}/{len(c['results'])} front "
                             f"members verify against {c['pack_id']}.{c['class']}")
                    if accepted == len(c["results"]):
                        st.success(label)
                    elif accepted > 0:
                        st.warning(label)
                    else:
                        st.error(label)
                    if mode == "strict" and accepted < len(c["results"]):
                        st.info(
                            f"**Strict compliance mode.** {c['pack_id']}.{c['class']} omits "
                            "one or more deemed-to-satisfy rules this model cannot source "
                            "(e.g. the SCM cap) — those are UNKNOWN, and strict mode "
                            "counts UNKNOWN as a violation. Zero (or few) verified-PASS "
                            "members here may reflect that ceiling, not a broken search — "
                            "check \"Allow UNKNOWN rules\" above to see members passing "
                            "every rule this pack actually sources."
                        )

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
                pk1.button(f"Max strength · {nsga_out['strength'][i_s]:.0f} MPa", key="pick_str",
                           on_click=load_mix_into, args=("A", list(nsga_out["mixes"][i_s])))
                pk2.button(f"Min carbon · {nsga_out['carbon'][i_c]:.0f} kg CO₂", key="pick_carb",
                           on_click=load_mix_into, args=("A", list(nsga_out["mixes"][i_c])))
                pk3.button(f"Min cost · ${nsga_out['cost'][i_m]:.0f}", key="pick_cost",
                           on_click=load_mix_into, args=("A", list(nsga_out["mixes"][i_m])))

                st.dataframe(
                    pd.DataFrame({"Strength": nsga_out["strength"], "Carbon": nsga_out["carbon"], "Cost": nsga_out["cost"]}).round(1),
                    use_container_width=True, height=240)

                st.subheader("Pareto front heatmap")
                fh = pd.DataFrame(nsga_out["mixes"], columns=param_names)
                hm = px.imshow(fh.T, labels=dict(x="Front mix", y="Parameter", color="Value"),
                               color_continuous_scale="Viridis", template="plotly_dark")
                hm.update_layout(height=420)
                st.plotly_chart(hm, use_container_width=True)
            else:
                with st.container(border=True):
                    st.markdown(f"**Results**\n\n{RESULTS_PLACEHOLDER_TEXT} — pick an algorithm "
                                "and settings at left, then hit Run.")

        # ---- GA / SA: weighted live optimization ------------------------------
        else:
            if run_clicked:
                # Top-of-column placeholder for the completion banner + metric
                # cards -- created FIRST (so it renders at the TOP of the results
                # area) but filled in only after the run loop below finishes.
                summary_slot = st.empty()

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
                        history_metrics["carbon"].append(carbon_for_mode(best_d, use_advanced_chemistry, **carbon_kwargs))
                        history_metrics["cost"].append(calculate_mix_cost(best_d, st.session_state.costs))

                        for ind in optimizer.population:
                            d = {k: v for k, v in zip(param_names, ind)}
                            all_pareto_points.append({
                                "Strength": predictor.predict(ind),
                                "Carbon": carbon_for_mode(d, use_advanced_chemistry, **carbon_kwargs),
                                "Cost": calculate_mix_cost(d, st.session_state.costs),
                                "Mix": "<br>".join([f"{param_names[i]}: {ind[i]:.1f}" for i in range(8)])
                            })

                        progress_bar.progress((g + 1) / total_steps)
                        live_status.markdown(
                            f"**Generation {g + 1}/{total_steps}** — best: "
                            f"{history_metrics['strength'][-1]:.1f} MPa · "
                            f"{history_metrics['carbon'][-1]:.1f} kg CO₂/m³ · "
                            f"${history_metrics['cost'][-1]:.2f}/m³  \n"
                            f"*(internal weighted score: {stats['best_fitness']:.2f})*"
                        )

                        # Convergence & Diversity Plot
                        conv_fig = go.Figure()
                        conv_fig.add_trace(go.Scatter(y=history_best, mode="lines+markers", name="Best (internal score)", line=dict(color="#00E676", width=3)))
                        conv_fig.add_trace(go.Scatter(y=history_avg, mode="lines", name="Avg (internal score)", line=dict(color="#2979FF", dash="dash")))
                        conv_fig.add_trace(go.Scatter(y=history_diversity, mode="lines", name="Gen Diversity", line=dict(color="#FFB300"), yaxis="y2"))
                        conv_fig.update_layout(
                            template="plotly_dark", title="GA Performance & Genetic Diversity",
                            xaxis_title="Generation", yaxis_title="Internal Score",
                            yaxis2=dict(title="Diversity (Std Dev)", overlaying="y", side="right"),
                            height=350, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.1)
                        )
                        convergence_placeholder.plotly_chart(conv_fig, use_container_width=True)

                        # Gene Distribution Plot -- explicit go.Violin traces (one
                        # per parameter) so the render is unambiguously a violin
                        # (kernel-density shape + inner box), matching the essay's
                        # "Violin Plots" wording below.
                        pop_df = pd.DataFrame(optimizer.population, columns=[p.replace('_', ' ').title() for p in param_names])
                        gene_fig = go.Figure()
                        palette = px.colors.qualitative.Plotly
                        for gi, gcol in enumerate(pop_df.columns):
                            gene_fig.add_trace(go.Violin(
                                y=pop_df[gcol], name=gcol, box_visible=True, meanline_visible=True,
                                points=False, line_color=palette[gi % len(palette)],
                            ))
                        gene_fig.update_layout(
                            template="plotly_dark", title="Population Gene Pool Distribution (Violin Plots)",
                            height=350, showlegend=False, xaxis_title="", yaxis_title="Mass (kg/m³)",
                            margin=dict(l=10, r=10, t=40, b=10),
                        )
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
                        history_metrics["carbon"].append(carbon_for_mode(best_d, use_advanced_chemistry, **carbon_kwargs))
                        history_metrics["cost"].append(calculate_mix_cost(best_d, st.session_state.costs))

                        for sol in [sa.current, sa.best]:
                            d = {k: v for k, v in zip(param_names, sol)}
                            all_pareto_points.append({
                                "Strength": predictor.predict(sol),
                                "Carbon": carbon_for_mode(d, use_advanced_chemistry, **carbon_kwargs),
                                "Cost": calculate_mix_cost(d, st.session_state.costs),
                                "Mix": "<br>".join([f"{param_names[i]}: {sol[i]:.1f}" for i in range(8)])
                            })

                        progress_bar.progress((step + 1) / total_steps)
                        live_status.markdown(
                            f"**Step {step + 1}/{total_steps}** (T={stats['temperature']:.2f}) — best: "
                            f"{history_metrics['strength'][-1]:.1f} MPa · "
                            f"{history_metrics['carbon'][-1]:.1f} kg CO₂/m³ · "
                            f"${history_metrics['cost'][-1]:.2f}/m³  \n"
                            f"*(internal weighted score: {stats['best_fitness']:.2f})*"
                        )

                        conv_fig = go.Figure()
                        conv_fig.add_trace(go.Scatter(y=history_best, mode="lines+markers", name="Global Best", line=dict(color="#00E676")))
                        conv_fig.add_trace(go.Scatter(y=history_avg, mode="lines", name="Current Temp Sol", line=dict(color="#FF5722", dash="dot")))
                        conv_fig.update_layout(template="plotly_dark", title=f"SA Trace (T={stats['temperature']:.2f})", xaxis_title="Step", yaxis_title="Internal Score", height=350)
                        convergence_placeholder.plotly_chart(conv_fig, use_container_width=True)

                        pareto_df = pd.DataFrame(all_pareto_points[-100:])
                        pareto_fig = px.scatter_3d(pareto_df, x="Strength", y="Carbon", z="Cost", color="Strength", hover_data=["Mix"], template="plotly_dark", title="Annealing trajectory (evaluated mixes)")
                        pareto_fig.update_layout(height=650)
                        pareto_placeholder.plotly_chart(pareto_fig, use_container_width=True)

                        if stats["temperature"] < 1e-6:
                            break

                    final_pop = np.array([sa.best])
                    best_ind = sa.best

                # ---- Completion summary: fills the top-of-column placeholder,
                # so it lands at/near the TOP of the results area rather than
                # mid-stream below the charts (R8.6 WP-U3 #3). The headline row
                # is the three physical metrics; the internal score, if shown at
                # all, is captioned rather than presented as a bare number.
                with summary_slot.container(border=True):
                    st.success(f"Optimization complete — {len(history_best)} steps run.")
                    b_col1, b_col2, b_col3 = st.columns(3)
                    with b_col1:
                        st.metric("Best Strength", f"{history_metrics['strength'][-1]:.1f} MPa")
                    with b_col2:
                        st.metric("Best Carbon", f"{history_metrics['carbon'][-1]:.1f} kg/m³")
                    with b_col3:
                        st.metric("Best Cost", f"${history_metrics['cost'][-1]:.2f}/m³")
                    st.caption(
                        f"Internal weighted score (higher is better): {history_best[-1]:.2f} "
                        "— not a physical quantity, only used to rank candidates during the search."
                    )

                # Metrics Evolution Plot -- three stacked subplots sharing the x-axis
                # instead of a triple-y-axis figure (R8.6 WP-U3 #5: the old third
                # axis collided with the second and clipped a tick).
                metrics_fig = make_subplots(
                    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                    subplot_titles=("Strength (MPa)", "Carbon (kg/m³)", "Cost ($/m³)"),
                )
                metrics_fig.add_trace(go.Scatter(y=history_metrics["strength"], name="Strength (MPa)", line=dict(color="#00E676")), row=1, col=1)
                metrics_fig.add_trace(go.Scatter(y=history_metrics["carbon"], name="Carbon (kg/m³)", line=dict(color="#FFB300")), row=2, col=1)
                metrics_fig.add_trace(go.Scatter(y=history_metrics["cost"], name="Cost ($/m³)", line=dict(color="#E91E63")), row=3, col=1)
                metrics_fig.update_xaxes(title_text="Generation/Step", row=3, col=1)
                metrics_fig.update_layout(
                    template="plotly_dark", title="Evolution of Best Solution Metrics",
                    height=560, showlegend=False, margin=dict(l=10, r=10, t=60, b=10),
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
            else:
                with st.container(border=True):
                    st.markdown(f"**Results**\n\n{RESULTS_PLACEHOLDER_TEXT} — pick an algorithm "
                                "and settings at left, then hit Run.")

    # R8.6 WP-U3 #7: essays moved into a collapsed expander, text verbatim.
    with st.expander("How the optimizers work (Philosophical Note on Stochastic Search)", expanded=False):
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
