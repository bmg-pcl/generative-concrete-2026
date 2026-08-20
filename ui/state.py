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
from src.generative_ga import PARAM_NAMES
from src.chemistry_simple import UNIT_COSTS, CARBON_FACTORS
from src.chemistry_advanced import GRID_EF
from src.exotics import EXOTIC_ADMIXTURES
from src.materials import slider_specs_view

# Mix slider specs: (param, label, min, max) in PARAM_NAMES order. R7.4b: the seven
# core materials' bounds live in data/materials.json (single UI authority — edit the
# registry, not this file); "age" is a design condition, not a material, so it has no
# registry record and is special-cased here.
_AGE_SLIDER = ("age", "Age (days)", 1, 365)
_slider_fields = slider_specs_view()
SLIDER_SPECS = [
    (p, _slider_fields[p]["label"], _slider_fields[p]["min"], _slider_fields[p]["max"])
    if p in _slider_fields else _AGE_SLIDER
    for p in PARAM_NAMES
]
DEFAULT_MIX_A = [300, 0, 0, 180, 0, 1000, 800, 28]
DEFAULT_MIX_B = [300, 100, 50, 160, 5, 1000, 800, 28]

# R7.4a: the Config tab's own controls, persisted in the session file. Same keyed-
# widget pattern as the mix sliders and cf_<mat> factors: the widget key IS the live
# value (no `value=` passed alongside `key=`), init_session_state() seeds the
# default via setdefault, and apply_session() writes the keys directly — legal
# because session import runs from the sidebar, which renders before the Config tab
# instantiates these widgets (see R4.1/R5.1 for the underlying Streamlit rule).
CONFIG_DEFAULTS = {
    "cfg_chemistry_mode": "Simple (Linear)",
    "cfg_transport_km": 0,
    "cfg_cement_source": "OPC (Portland)",
    "cfg_kiln_fuel": "Default (unspecified mix)",
    "cfg_electricity": sorted(GRID_EF.keys())[0],
    "cfg_capture_rate": 0.0,
    "cfg_robust": True,
    "cfg_fix_age": True,
    "cfg_design_age": 28,
    "cfg_exotic_strength": False,
    # R8.0 WP-A A2: batched-vs-placed waste factor (default 0.0 -- inert). Added
    # HERE only: the ui_config export/import comprehension picks it up automatically
    # (see export_session below) -- no SESSION_FIELDS edit, no SESSION_VERSION bump.
    "cfg_waste_factor": 0.0,
}


def load_mix_into(slot: str, mix_vec, state=None):
    """Write a mix vector into the keyed sliders for slot 'A'/'B', clamped to range.

    Must be called from a callback (on_click/on_change) or before the sliders
    instantiate, so the write precedes widget instantiation on the next run.
    `state` defaults to st.session_state; tests may pass any mutable mapping."""
    state = st.session_state if state is None else state
    for (p, _, lo, hi), v in zip(SLIDER_SPECS, mix_vec):
        state[f"{p}_{slot}"] = int(np.clip(round(float(v)), lo, hi))


def current_mix(slot: str, state=None) -> np.ndarray:
    """Assemble the mix vector for slot 'A'/'B' from its keyed sliders."""
    state = st.session_state if state is None else state
    return np.array([state[f"{p}_{slot}"] for p, *_ in SLIDER_SPECS])


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
    # The emission-factor editors are keyed widgets (cf_<mat>); their keys are the
    # live value and carbon_factors is re-derived from them each run (same pattern
    # as the mix sliders — one source of truth, programmatic loads write the keys).
    for mat, val in st.session_state.carbon_factors.items():
        st.session_state.setdefault(f"cf_{mat}", float(val))
    if "exotic_a" not in st.session_state:
        st.session_state.exotic_a = {k: v["default"] for k, v in EXOTIC_ADMIXTURES.items()}
    if "exotic_b" not in st.session_state:
        st.session_state.exotic_b = {k: v["default"] for k, v in EXOTIC_ADMIXTURES.items()}
    if "epds" not in st.session_state:
        st.session_state.epds = {}   # attached supplier EPDs {material: record} (R6.2)
    for key, default in CONFIG_DEFAULTS.items():
        st.session_state.setdefault(key, default)


# Every field the session file carries (besides "version"). export_session writes
# exactly these; apply_session consumes exactly these. Add a field HERE and both
# sides pick it up — the round-trip tests enforce the symmetry.
SESSION_FIELDS = ("mix_a", "mix_b", "costs", "carbon_factors", "exotic_a", "exotic_b", "epds", "ui_config")
SESSION_VERSION = 4   # v4 adds ui_config (the Config-tab controls); v1-v3 files simply keep Config defaults


def export_session(state=None) -> dict:
    """The complete session as a plain dict (the export file's schema)."""
    state = st.session_state if state is None else state
    return {
        "version": SESSION_VERSION,
        "mix_a": current_mix("A", state).tolist(),
        "mix_b": current_mix("B", state).tolist(),
        "costs": dict(state["costs"]),
        "carbon_factors": dict(state["carbon_factors"]),   # was dropped in v1 (regression)
        "exotic_a": dict(state["exotic_a"]),
        "exotic_b": dict(state["exotic_b"]),
        "epds": dict(state["epds"]) if "epds" in state else {},
        # R7.4a: the Config tab's own controls (transport, clinker source, robust/age
        # toggles, ...) — previously lost on every export (the R3.4 deferral). Named
        # "ui_config", NOT "config": the CLI (src/cli.py) already uses a top-level
        # "config" key with a DIFFERENT, semantic schema (advanced/robust/age/...)
        # and rejects unknown keys there — colliding names would break
        # `python -m src.cli ... --config <exported-session.json>`.
        "ui_config": {k: state[k] for k in CONFIG_DEFAULTS if k in state},
    }


def apply_session(data: dict, state=None):
    """Write an imported session into state. Must run before widgets instantiate
    (the sidebar runs first, so calling it there is safe).

    Writes BOTH the plain dicts and the backing widget keys: the mix sliders
    (cement_A, ...), the emission-factor editors (cf_<mat>), and the Config-tab
    controls (cfg_<...>) are all keyed widgets, so an import that only replaced the
    dicts would be silently overwritten by the widgets' retained state on the very
    next rerun."""
    state = st.session_state if state is None else state
    load_mix_into("A", data["mix_a"], state)
    load_mix_into("B", data["mix_b"], state)
    state["costs"] = data["costs"]
    if "carbon_factors" in data:   # v2; v1 files simply keep the defaults
        state["carbon_factors"] = data["carbon_factors"]
        for mat, val in data["carbon_factors"].items():
            state[f"cf_{mat}"] = float(val)
    if "exotic_a" in data:
        state["exotic_a"] = data["exotic_a"]
        state["exotic_b"] = data["exotic_b"]
    if "epds" in data:   # v3; older files simply carry no EPD attachments
        state["epds"] = data["epds"]
    if "ui_config" in data:   # v4; v1-v3 files simply keep the Config-tab defaults
        for k, v in data["ui_config"].items():
            state[k] = v


def get_state_json() -> str:
    return json.dumps(export_session())
