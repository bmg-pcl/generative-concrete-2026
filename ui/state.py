"""Shared session-state primitives and cached resources for the app.

The mix sliders are keyed widgets (f"{param}_A"/f"{param}_B") whose values are the
single source of truth; the mix vector is derived from those keys each run, and every
programmatic load writes the keys via a callback so the write lands before the widgets
instantiate on the next run.
"""
import json

import numpy as np
import streamlit as st

from src.models import StrengthPredictor
from src.bayesian import BayesFlowExplorer
from src.data_fetcher import load_data
from src.chemistry_simple import UNIT_COSTS, CARBON_FACTORS
from src.exotics import EXOTIC_ADMIXTURES

# Mix slider specs: (param, label, min, max) in PARAM_NAMES order.
SLIDER_SPECS = [
    ("cement", "Cement", 100, 550),
    ("slag", "Slag", 0, 360),
    ("ash", "Fly Ash", 0, 200),
    ("water", "Water", 120, 250),
    ("superplasticizer", "Superplasticizer", 0, 30),
    ("coarse_agg", "Coarse Agg", 700, 1150),
    ("fine_agg", "Fine Agg", 550, 1000),
    ("age", "Age (days)", 1, 365),
]
DEFAULT_MIX_A = [300, 0, 0, 180, 0, 1000, 800, 28]
DEFAULT_MIX_B = [300, 100, 50, 160, 5, 1000, 800, 28]


def load_mix_into(slot: str, mix_vec):
    """Write a mix vector into the keyed sliders for slot 'A'/'B', clamped to range.

    Must be called from a callback (on_click/on_change) or before the sliders
    instantiate, so the write precedes widget instantiation on the next run."""
    for (p, _, lo, hi), v in zip(SLIDER_SPECS, mix_vec):
        st.session_state[f"{p}_{slot}"] = int(np.clip(round(float(v)), lo, hi))


def current_mix(slot: str) -> np.ndarray:
    """Assemble the mix vector for slot 'A'/'B' from its keyed sliders."""
    return np.array([st.session_state[f"{p}_{slot}"] for p, *_ in SLIDER_SPECS])


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


@st.cache_resource
def load_resources():
    predictor = StrengthPredictor()
    try:
        predictor.predict(np.zeros((1, 8)))
    except Exception:
        predictor.train()
    bayesian = BayesFlowExplorer()
    return predictor, bayesian


def init_session_state():
    """Initialise keyed slider values and the shared dict-state (idempotent)."""
    for slot, default in (("A", DEFAULT_MIX_A), ("B", DEFAULT_MIX_B)):
        for (p, _, _lo, _hi), v in zip(SLIDER_SPECS, default):
            st.session_state.setdefault(f"{p}_{slot}", int(v))
    if "costs" not in st.session_state:
        st.session_state.costs = UNIT_COSTS.copy()
    if "carbon_factors" not in st.session_state:
        st.session_state.carbon_factors = CARBON_FACTORS.copy()
    if "exotic_a" not in st.session_state:
        st.session_state.exotic_a = {k: v["default"] for k, v in EXOTIC_ADMIXTURES.items()}
    if "exotic_b" not in st.session_state:
        st.session_state.exotic_b = {k: v["default"] for k, v in EXOTIC_ADMIXTURES.items()}


def get_state_json() -> str:
    state = {
        "version": 2,
        "mix_a": current_mix("A").tolist(),
        "mix_b": current_mix("B").tolist(),
        "costs": st.session_state.costs,
        "carbon_factors": st.session_state.carbon_factors,   # was dropped in v1 (regression)
        "exotic_a": st.session_state.exotic_a,
        "exotic_b": st.session_state.exotic_b,
    }
    return json.dumps(state)
