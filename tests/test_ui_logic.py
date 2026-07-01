"""
Tests for the extracted UI logic (src/ui_logic.py). These verify the *coherence*
guarantees the app relies on, without needing Streamlit.
"""
import numpy as np
import pandas as pd
import pytest

from src.ui_logic import (
    PARAM_NAMES,
    carbon_for_mode,
    compute_metrics,
    batch_metrics,
    scalarized_fitness,
    recommend_recipe,
    validate_lab_csv,
    validate_session_state,
    mix_dict,
)
from src.chemistry_simple import calculate_embodied_carbon, calculate_mix_cost
from src.chemistry_advanced import embodied_carbon_advanced
from src.exotics import exotic_strength_delta
from src.models import StrengthPredictor

MIX = [350, 100, 0, 175, 5, 1000, 750, 28]
COSTS = {"cement": 0.15, "slag": 0.08, "ash": 0.05, "water": 0.002,
         "superplasticizer": 2.5, "coarse_agg": 0.03, "fine_agg": 0.04}


@pytest.fixture(scope="module")
def predictor():
    return StrengthPredictor()


def _no_exotics():
    from src.exotics import EXOTIC_ADMIXTURES
    return {k: 0 for k in EXOTIC_ADMIXTURES}


def test_carbon_for_mode_matches_underlying():
    d = mix_dict(MIX)
    assert carbon_for_mode(d, advanced=False) == calculate_embodied_carbon(d)
    assert carbon_for_mode(d, advanced=True) == embodied_carbon_advanced(d)


def test_metrics_respect_chemistry_mode(predictor):
    simple = compute_metrics(MIX, _no_exotics(), COSTS, predictor, advanced=False)
    advanced = compute_metrics(MIX, _no_exotics(), COSTS, predictor, advanced=True)
    assert simple["carbon"] != advanced["carbon"]
    # Strength is identical regardless of carbon tier.
    assert simple["strength"] == advanced["strength"]


def test_metrics_exotic_strength_switch(predictor):
    exotic = _no_exotics(); exotic["silica_fume"] = 50
    off = compute_metrics(MIX, exotic, COSTS, predictor, exotic_strength=False)
    on = compute_metrics(MIX, exotic, COSTS, predictor, exotic_strength=True)
    assert off["exotic_strength"] == 0.0
    assert on["exotic_strength"] == exotic_strength_delta(exotic, enabled=True)
    assert on["strength"] == off["strength"] + on["exotic_strength"]
    # Carbon/cost include the exotic in BOTH modes.
    assert off["carbon"] > compute_metrics(MIX, _no_exotics(), COSTS, predictor)["carbon"]


def test_batch_metrics_shapes(predictor):
    samples = np.array([MIX, MIX, MIX], dtype=float)
    m = batch_metrics(samples, COSTS, predictor)
    assert m["strength"].shape == (3,)
    assert m["carbon"].shape == (3,) and m["cost"].shape == (3,)


def test_scalarized_fitness_formula(predictor):
    f = scalarized_fitness(MIX, COSTS, predictor, 1.0, 0.05, 0.5, advanced=False)
    d = mix_dict(MIX)
    expected = (1.0 * predictor.predict(np.asarray(MIX, float))
                - 0.05 * calculate_embodied_carbon(d)
                - 0.5 * calculate_mix_cost(d, COSTS))
    assert f == pytest.approx(expected)


def test_recommend_recipe_hits_target_ga():
    from src.bayesian import BayesFlowExplorer
    explorer = BayesFlowExplorer()
    rec = recommend_recipe(explorer, 45.0, method="ga", costs=COSTS)
    assert abs(rec["strength"] - 45.0) < 4.0
    assert set(rec["params"]) == set(PARAM_NAMES)
    assert rec["carbon"] > 0 and rec["cost"] > 0


def test_validate_lab_csv():
    good = pd.DataFrame([dict(zip(PARAM_NAMES + ["strength"], MIX + [40]))])
    assert validate_lab_csv(good) is None
    assert "Missing" in validate_lab_csv(good.drop(columns=["strength"]))
    bad = good.copy(); bad["cement"] = "oops"
    assert "Non-numeric" in validate_lab_csv(bad)
    assert "no rows" in validate_lab_csv(good.iloc[0:0])


def test_validate_session_state():
    ok = {"mix_a": MIX, "mix_b": MIX, "costs": COSTS}
    assert validate_session_state(ok) is None
    assert "missing" in validate_session_state({"mix_a": MIX}).lower()
    assert "values" in validate_session_state({"mix_a": [1, 2], "mix_b": MIX, "costs": COSTS})
    assert validate_session_state([1, 2, 3]) is not None
