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
    pareto_front_mask,
    compute_metrics,
    batch_metrics,
    scalarized_fitness,
    recommend_recipe,
    validate_lab_csv,
    validate_session_state,
    mix_dict,
    tensile_estimate,
    carbon_breakdown,
    mix_ticket,
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


def test_carbon_for_mode_transport_and_factor_overrides():
    d = mix_dict(MIX)
    base = carbon_for_mode(d, advanced=False)
    assert carbon_for_mode(d, advanced=False, transport_km=500) > base   # transport adds
    zero_factors = {k: 0.0 for k in ["cement", "slag", "ash", "water",
                                     "superplasticizer", "coarse_agg", "fine_agg"]}
    assert carbon_for_mode(d, advanced=False, factors=zero_factors) == 0.0  # override applies


# --- R7.5 WP-5: transport mass consistency ---------------------------------------

def test_carbon_for_mode_exotic_raises_transport_term():
    """A mix with exotics dosed shows a HIGHER transport term than the same mix
    without -- exact expected delta: 100 kg/m3 x 500 km x 0.1/1000 = 5.0 kg CO2."""
    d = mix_dict(MIX)
    exotic = {"silica_fume": 100.0}
    expected_delta = (100.0 / 1000.0) * 500.0 * 0.1
    assert expected_delta == pytest.approx(5.0)
    for advanced in (False, True):
        base = carbon_for_mode(d, advanced=advanced, transport_km=500)
        dosed = carbon_for_mode(d, advanced=advanced, transport_km=500, exotic=exotic)
        assert dosed - base == pytest.approx(expected_delta)


def test_carbon_for_mode_exotic_default_is_bit_identical():
    d = mix_dict(MIX)
    assert (carbon_for_mode(d, advanced=False, transport_km=500)
            == carbon_for_mode(d, advanced=False, transport_km=500, exotic=None))
    assert (carbon_for_mode(d, advanced=True, transport_km=500)
            == carbon_for_mode(d, advanced=True, transport_km=500, exotic=None))


def test_compute_metrics_carbon_rises_with_exotic_transport_mass():
    """Dosing exotics with a nonzero transport leg must raise the carbon metric by
    at least the transport delta on top of the exotics' own carbon factor -- the
    exotic mass must not ship for free."""
    predictor = StrengthPredictor()
    exotic_off = _no_exotics()
    exotic_on = _no_exotics()
    exotic_on["silica_fume"] = 100.0
    carbon_kwargs = {"transport_km": 500.0}
    off = compute_metrics(MIX, exotic_off, COSTS, predictor, carbon_kwargs=carbon_kwargs)
    on = compute_metrics(MIX, exotic_on, COSTS, predictor, carbon_kwargs=carbon_kwargs)
    from src.exotics import exotic_carbon
    own_factor_delta = exotic_carbon(exotic_on) - exotic_carbon(exotic_off)
    transport_delta = (100.0 / 1000.0) * 500.0 * 0.1
    assert on["carbon"] - off["carbon"] == pytest.approx(own_factor_delta + transport_delta)


def test_metrics_respect_chemistry_mode(predictor):
    simple = compute_metrics(MIX, _no_exotics(), COSTS, predictor, advanced=False)
    advanced = compute_metrics(MIX, _no_exotics(), COSTS, predictor, advanced=True)
    assert simple["carbon"] != advanced["carbon"]
    # Strength is identical regardless of carbon tier.
    assert simple["strength"] == advanced["strength"]


def test_metrics_exotic_strength_switch(predictor):
    exotic = _no_exotics()
    exotic["silica_fume"] = 50
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
    assert m["novelty"].shape == (3,)


def test_metrics_include_interval_and_support(predictor):
    m = compute_metrics(MIX, _no_exotics(), COSTS, predictor)
    assert m["interval_lo"] < m["interval_hi"]
    assert isinstance(m["in_support"], bool)
    assert m["novelty"] >= 0


def test_scalarized_fitness_formula(predictor):
    f = scalarized_fitness(MIX, COSTS, predictor, 1.0, 0.05, 0.5, advanced=False)
    d = mix_dict(MIX)
    expected = (1.0 * predictor.predict(np.asarray(MIX, float))
                - 0.05 * calculate_embodied_carbon(d)
                - 0.5 * calculate_mix_cost(d, COSTS))
    assert f == pytest.approx(expected)


def test_recommend_recipe_hits_target_ga():
    from src.bayesian import BayesFlowExplorer
    np.random.seed(0)  # deterministic: the GA is stochastic
    explorer = BayesFlowExplorer()
    rec = recommend_recipe(explorer, 45.0, method="ga", costs=COSTS)
    assert abs(rec["strength"] - 45.0) < 4.0
    assert set(rec["params"]) == set(PARAM_NAMES)
    assert rec["carbon"] > 0 and rec["cost"] > 0


def test_robust_recipe_in_support_and_meets_lower_bound():
    """R1.3 gate: a robust recipe is in-support and its guaranteed (lower-bound)
    strength meets the target."""
    from src.bayesian import BayesFlowExplorer
    np.random.seed(0)
    explorer = BayesFlowExplorer()
    rec = recommend_recipe(explorer, 40.0, method="ga", costs=COSTS, robust=True)
    assert rec["in_support"]
    assert rec["interval_lo"] >= 40.0 - 5.0


def test_scalarized_fitness_robust_uses_lower_bound(predictor):
    """Robust fitness should differ from mean fitness (uses the lower bound + OOS penalty)."""
    f_mean = scalarized_fitness(MIX, COSTS, predictor, 1.0, 0.05, 0.5, robust=False)
    f_rob = scalarized_fitness(MIX, COSTS, predictor, 1.0, 0.05, 0.5, robust=True)
    assert f_rob < f_mean  # lower bound < mean, so robust fitness is lower


def test_validate_lab_csv():
    good = pd.DataFrame([dict(zip(PARAM_NAMES + ["strength"], MIX + [40]))])
    assert validate_lab_csv(good) is None
    assert "Missing required" in validate_lab_csv(good.drop(columns=["strength"]))
    bad = good.copy()
    bad["cement"] = "oops"
    assert validate_lab_csv(bad) is not None
    assert "no rows" in validate_lab_csv(good.iloc[0:0])
    # An all-NaN required column must be rejected (would poison retraining).
    nan_strength = pd.DataFrame([dict(zip(PARAM_NAMES + ["strength"], MIX + [np.nan]))])
    assert validate_lab_csv(nan_strength) is not None


def test_pareto_front_mask():
    # Objectives: max strength, min carbon, min cost.
    # A: 40/200/100  B: 50/200/100 (dominates A)  C: 45/150/120 (non-dominated vs B)
    # D: 50/210/110 (dominated by B)
    strength = [40, 50, 45, 50]
    carbon = [200, 200, 150, 210]
    cost = [100, 100, 120, 110]
    mask = pareto_front_mask(strength, carbon, cost)
    assert list(mask) == [False, True, True, False]
    # A single point is always on its own front.
    assert list(pareto_front_mask([30], [100], [50])) == [True]


def test_validate_session_state():
    ok = {"mix_a": MIX, "mix_b": MIX, "costs": COSTS}
    assert validate_session_state(ok) is None
    assert "missing" in validate_session_state({"mix_a": MIX}).lower()
    assert "values" in validate_session_state({"mix_a": [1, 2], "mix_b": MIX, "costs": COSTS})
    assert validate_session_state([1, 2, 3]) is not None


def test_tensile_estimate_ec2_branches():
    # Below/above the 50 MPa branch switch, both EC2 formulae.
    assert tensile_estimate(30.0) == pytest.approx(0.30 * 30.0 ** (2.0 / 3.0))
    assert tensile_estimate(60.0) == pytest.approx(2.12 * np.log(1.0 + (60.0 + 8.0) / 10.0))
    # Continuous across the 50 MPa branch switch (first branch at 50.0, second just above).
    assert tensile_estimate(50.0) == pytest.approx(tensile_estimate(50.01), abs=0.05)
    assert tensile_estimate(70.0) > tensile_estimate(40.0) > tensile_estimate(20.0)
    # Non-negative and clamps a nonsensical negative input.
    assert tensile_estimate(-5.0) == 0.0


def test_carbon_breakdown_sums_to_carbon_for_mode():
    d = mix_dict(MIX)
    # Simple mode, with a transport leg: the per-source breakdown must sum exactly
    # to the single displayed carbon figure (the mix ticket relies on this).
    bd = carbon_breakdown(d, advanced=False, transport_km=50.0)
    assert sum(bd.values()) == pytest.approx(
        carbon_for_mode(d, advanced=False, transport_km=50.0)
    )
    assert "transport" in bd


def test_carbon_breakdown_sums_to_carbon_for_mode_with_exotic():
    """R7.5 WP-5: the invariant holds WITH an exotic dosing dict too, in both
    tiers -- the "transport" entry, not a hidden extra term, carries the exotic
    mass."""
    d = mix_dict(MIX)
    exotic = {"silica_fume": 100.0}
    for advanced in (False, True):
        bd = carbon_breakdown(d, advanced=advanced, transport_km=50.0, exotic=exotic)
        assert sum(bd.values()) == pytest.approx(
            carbon_for_mode(d, advanced=advanced, transport_km=50.0, exotic=exotic)
        )
        bd_no_exotic = carbon_breakdown(d, advanced=advanced, transport_km=50.0)
        assert bd["transport"] > bd_no_exotic["transport"]


def test_carbon_breakdown_reconciles_in_advanced_mode():
    d = mix_dict(MIX)
    # Advanced tier replaces the cement term with clinker chemistry; the breakdown
    # must still sum to the displayed advanced-tier carbon, incl. an LC3 clinker
    # source and a transport leg (the ticket's TOTAL row depends on this).
    bd = carbon_breakdown(d, advanced=True, transport_km=50.0, cement_type="LC3")
    assert sum(bd.values()) == pytest.approx(
        carbon_for_mode(d, advanced=True, transport_km=50.0, cement_type="LC3")
    )


def test_mix_ticket_total_reconciles_in_advanced_mode(predictor):
    d = mix_dict(MIX)
    m = compute_metrics(MIX, _no_exotics(), COSTS, predictor, advanced=True,
                        carbon_kwargs={"transport_km": 50.0, "cement_type": "LC3"})
    config = {"advanced": True, "transport_km": 50.0, "cement_type": "LC3",
              "factors": None, "costs": COSTS, "robust": True,
              "timestamp": "2026-07-01T00:00:00+00:00"}
    csv = mix_ticket(d, m, config)
    total_line = next(line for line in csv.splitlines() if line.startswith("carbon_kgCO2,TOTAL,"))
    ticket_total = float(total_line.split(",")[2])
    assert ticket_total == pytest.approx(
        carbon_for_mode(d, advanced=True, transport_km=50.0, cement_type="LC3"), abs=0.05
    )


def test_mix_ticket_is_parseable_and_balanced(predictor):
    d = mix_dict(MIX)
    exotic = _no_exotics()
    m = compute_metrics(MIX, exotic, COSTS, predictor)
    config = {"advanced": False, "transport_km": 0.0, "cement_type": "OPC",
              "factors": None, "costs": COSTS, "robust": True, "timestamp": "2026-07-01T00:00:00+00:00"}
    csv = mix_ticket(d, m, config)
    lines = csv.splitlines()
    assert lines[0] == "section,key,value"
    # Every mix parameter appears as a row.
    for p in PARAM_NAMES:
        assert any(line.startswith(f"mix,{p},") for line in lines)
    # The carbon TOTAL row equals the displayed carbon (breakdown reconciles).
    total_line = next(line for line in lines if line.startswith("carbon_kgCO2,TOTAL,"))
    ticket_total = float(total_line.split(",")[2])
    assert ticket_total == pytest.approx(
        carbon_for_mode(d, advanced=False, transport_km=0.0), abs=0.05
    )
    assert any("disclaimer" in line for line in lines)
