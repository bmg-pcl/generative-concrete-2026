"""Calibration tab — retrain on lab data, active learning, accuracy history."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_fetcher import append_experimental_results
from src.models import load_metrics_history
from src.ui_logic import validate_lab_csv
from ui.context import AppContext
from ui.state import load_resources


def render_calibration(ctx: AppContext):
    predictor = ctx.predictor
    bayesian = ctx.bayesian

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
                load_resources.clear()   # rebuild the predictor + explorer
                st.cache_data.clear()     # invalidate cached samples/recipes/presets
                st.success("Model calibrated with actual field results!")
                st.info(
                    "The amortized BayesFlow flow is now **stale** — it was trained against the "
                    "previous model — so Inverse Design falls back to the GA designer. Retrain the "
                    "flow with `python -m src.amortized` to re-enable it.",
                    icon="ℹ️",
                )
                st.balloons()

    st.divider()
    st.subheader("Active learning: which lab tests to run next")
    st.caption(
        "Rather than testing at random, this ranks candidate mixes by how much running "
        "them would sharpen the model — mixes near your target, in under-sampled regions "
        "(high novelty), with wide prediction intervals. Running the top rows and feeding "
        "the results back above shrinks the model's blind spots fastest."
    )
    al_c1, al_c2 = st.columns([1, 1])
    with al_c1:
        al_target = st.number_input("Target strength for suggestions (MPa)", 10, 100, 45, key="al_target")
    with al_c2:
        al_n = st.number_input("Number of tests to suggest", 3, 20, 5, key="al_n")
    if st.button("Suggest tests"):
        with st.spinner("Ranking informative candidate mixes…"):
            st.session_state.al_suggestions = bayesian.suggest_tests(
                float(al_target), n_tests=int(al_n)
            )
    if st.session_state.get("al_suggestions") is not None:
        sugg = st.session_state.al_suggestions
        st.dataframe(sugg, use_container_width=True, height=260)
        st.download_button(
            "Download suggested tests (CSV)", data=sugg.to_csv(index=False),
            file_name="suggested_lab_tests.csv", mime="text/csv",
        )
        st.caption("merit = interval half-width × min(novelty, 3) / (1 + |strength − target|)")

    hist = load_metrics_history()
    if hist:
        st.divider()
        st.subheader("Model accuracy over retrains")
        hist_df = pd.DataFrame(hist)
        acc_fig = go.Figure()
        acc_fig.add_trace(go.Scatter(y=hist_df["rmse"], mode="lines+markers",
                                     name="RMSE (MPa)", line=dict(color="#E91E63")))
        acc_fig.add_trace(go.Scatter(y=hist_df["r2"], mode="lines+markers",
                                     name="R²", line=dict(color="#00E676"), yaxis="y2"))
        acc_fig.update_layout(
            template="plotly_dark", height=300, margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Retrain #",
            yaxis=dict(title=dict(text="RMSE (MPa)", font=dict(color="#E91E63")), tickfont=dict(color="#E91E63")),
            yaxis2=dict(title=dict(text="R²", font=dict(color="#00E676")), tickfont=dict(color="#00E676"),
                        overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.2))
        st.plotly_chart(acc_fig, use_container_width=True)
        st.dataframe(hist_df, use_container_width=True, height=180)

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
