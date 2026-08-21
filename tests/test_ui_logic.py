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
    """R8.0 WP-A A1: WITH an exotic dosing dict, the breakdown sums to the
    DISPLAYED carbon -- carbon_for_mode(...) (which already carries the exotic
    transport mass, R7.5 WP-5) PLUS the exotics' own carbon factor via the new
    "exotics" line. Before A1 the ticket omitted that factor entirely."""
    from src.exotics import exotic_carbon
    d = mix_dict(MIX)
    exotic = {"silica_fume": 100.0}
    for advanced in (False, True):
        bd = carbon_breakdown(d, advanced=advanced, transport_km=50.0, exotic=exotic)
        assert sum(bd.values()) == pytest.approx(
            carbon_for_mode(d, advanced=advanced, transport_km=50.0, exotic=exotic)
            + exotic_carbon(exotic)
        )
        assert bd["exotics"] == pytest.approx(exotic_carbon(exotic))
        bd_no_exotic = carbon_breakdown(d, advanced=advanced, transport_km=50.0)
        assert bd["transport"] > bd_no_exotic["transport"]
        assert bd_no_exotic["exotics"] == 0.0  # default exotic=None is inert


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


# --- R8.0 WP-A A1: ticket reconciliation for a DOSED mix (the headline gate) ------

def test_mix_ticket_total_equals_displayed_carbon_for_dosed_mix(predictor):
    """The A1 headline gate: for a mix with exotics dosed, the ticket's
    carbon_kgCO2,TOTAL row must equal compute_metrics(...)["carbon"] -- the number
    actually displayed in the UI -- not just carbon_for_mode(...) alone (which omits
    the exotics' own carbon factor)."""
    d = mix_dict(MIX)
    exotic = _no_exotics()
    exotic["silica_fume"] = 80.0
    exotic["nano_silica"] = 5.0
    carbon_kwargs = {"transport_km": 50.0}
    m = compute_metrics(MIX, exotic, COSTS, predictor, carbon_kwargs=carbon_kwargs)
    config = {"advanced": False, "transport_km": 50.0, "cement_type": "OPC",
              "factors": None, "costs": COSTS, "robust": True,
              "timestamp": "2026-07-01T00:00:00+00:00"}
    csv = mix_ticket(d, m, config, exotic=exotic)
    total_line = next(line for line in csv.splitlines() if line.startswith("carbon_kgCO2,TOTAL,"))
    ticket_total = float(total_line.split(",")[2])
    assert ticket_total == pytest.approx(m["carbon"], abs=0.05)


def test_mix_ticket_omitting_exotic_undercounts_for_a_dosed_mix(predictor):
    """Documents the failure mode A1 fixes: forgetting to pass `exotic` to
    mix_ticket for a dosed mix makes the ticket total fall SHORT of the displayed
    carbon -- this is why every call site must pass the dosing dict it holds."""
    d = mix_dict(MIX)
    exotic = _no_exotics()
    exotic["silica_fume"] = 80.0
    m = compute_metrics(MIX, exotic, COSTS, predictor)
    config = {"advanced": False, "transport_km": 0.0, "cement_type": "OPC",
              "factors": None, "costs": COSTS, "robust": True,
              "timestamp": "2026-07-01T00:00:00+00:00"}
    csv = mix_ticket(d, m, config)  # exotic NOT passed -- the old, buggy call shape
    total_line = next(line for line in csv.splitlines() if line.startswith("carbon_kgCO2,TOTAL,"))
    ticket_total = float(total_line.split(",")[2])
    assert ticket_total < m["carbon"] - 0.01


def test_mix_ticket_exotic_none_is_bit_identical_to_before():
    """Default `exotic=None` on mix_ticket must not change any existing number."""
    d = mix_dict(MIX)
    predictor = StrengthPredictor()
    m = compute_metrics(MIX, _no_exotics(), COSTS, predictor)
    config = {"advanced": False, "transport_km": 0.0, "cement_type": "OPC",
              "factors": None, "costs": COSTS, "robust": True,
              "timestamp": "2026-07-01T00:00:00+00:00"}
    assert mix_ticket(d, m, config) == mix_ticket(d, m, config, exotic=None)


# --- R8.0 WP-A A2: waste factor (batched vs placed) -------------------------------

def test_compute_metrics_waste_factor_inert_at_default(predictor):
    exotic = _no_exotics()
    m0 = compute_metrics(MIX, exotic, COSTS, predictor)
    m_explicit = compute_metrics(MIX, exotic, COSTS, predictor, waste_factor=0.0)
    assert m0["carbon"] == m_explicit["carbon"]
    assert m0["carbon"] == m0["carbon_as_placed"]   # wf=0 => as-placed == batched
    assert m0["cost"] == m0["cost_as_placed"]


def test_compute_metrics_waste_factor_scales_as_placed_only(predictor):
    exotic = _no_exotics()
    base = compute_metrics(MIX, exotic, COSTS, predictor)
    wasted = compute_metrics(MIX, exotic, COSTS, predictor, waste_factor=0.05)
    # The batched (per-m3) figures never move -- optimizers keep optimizing these.
    assert wasted["carbon"] == base["carbon"]
    assert wasted["cost"] == base["cost"]
    # The as-placed figures scale by (1 + wf).
    assert wasted["carbon_as_placed"] == pytest.approx(base["carbon"] * 1.05)
    assert wasted["cost_as_placed"] == pytest.approx(base["cost"] * 1.05)


def test_mix_ticket_waste_factor_rows(predictor):
    d = mix_dict(MIX)
    exotic = _no_exotics()
    m = compute_metrics(MIX, exotic, COSTS, predictor, waste_factor=0.05)
    config = {"advanced": False, "transport_km": 0.0, "cement_type": "OPC",
              "factors": None, "costs": COSTS, "robust": True, "waste_factor": 0.05,
              "timestamp": "2026-07-01T00:00:00+00:00"}
    csv = mix_ticket(d, m, config)
    lines = csv.splitlines()
    assert any(line == "config,waste_factor,0.05" for line in lines)
    total_line = next(line for line in lines if line.startswith("carbon_kgCO2,TOTAL,"))
    placed_line = next(line for line in lines if line.startswith("carbon_kgCO2,TOTAL_as_placed,"))
    total = float(total_line.split(",")[2])
    placed = float(placed_line.split(",")[2])
    assert placed == pytest.approx(total * 1.05, abs=0.01)


def test_mix_ticket_waste_factor_inert_at_default(predictor):
    """wf=0 (the default -- whether via an absent key or an explicit 0.0) leaves
    TOTAL_as_placed bit-identical to TOTAL."""
    d = mix_dict(MIX)
    exotic = _no_exotics()
    m = compute_metrics(MIX, exotic, COSTS, predictor)
    config = {"advanced": False, "transport_km": 0.0, "cement_type": "OPC",
              "factors": None, "costs": COSTS, "robust": True,
              "timestamp": "2026-07-01T00:00:00+00:00"}
    csv = mix_ticket(d, m, config)
    lines = csv.splitlines()
    total_line = next(line for line in lines if line.startswith("carbon_kgCO2,TOTAL,"))
    placed_line = next(line for line in lines if line.startswith("carbon_kgCO2,TOTAL_as_placed,"))
    assert total_line.split(",")[2] == placed_line.split(",")[2]


# --- R8.0 WP-A A3: carbon-intensity KPI -------------------------------------------

def test_compute_metrics_carbon_intensity(predictor):
    exotic = _no_exotics()
    m = compute_metrics(MIX, exotic, COSTS, predictor)
    assert m["carbon_intensity"] == pytest.approx(m["carbon"] / max(m["strength"], 1.0))


def test_mix_ticket_carbon_intensity_row(predictor):
    d = mix_dict(MIX)
    exotic = _no_exotics()
    m = compute_metrics(MIX, exotic, COSTS, predictor)
    config = {"advanced": False, "transport_km": 0.0, "cement_type": "OPC",
              "factors": None, "costs": COSTS, "robust": True,
              "timestamp": "2026-07-01T00:00:00+00:00"}
    csv = mix_ticket(d, m, config)
    row = next(line for line in csv.splitlines()
               if line.startswith("prediction,carbon_intensity_kg_per_m3MPa_point,"))
    value = float(row.split(",")[2])
    assert value == pytest.approx(m["carbon_intensity"], abs=0.001)


# --- R8.0 WP-E: disclosure additions -----------------------------------------------
# Wave B wiring: carbon interval (D3), allocation+vintage (D1), compliance
# warnings (D2), thermal/carbonation (C1/C3), secondary maturity curing (C2,
# Decision 1), per-material transport (D4, Decision 2). Every gate here is
# additive-only: existing keys/rows are untouched at defaults.

DEFAULT_CONFIG = {"advanced": False, "transport_km": 0.0, "cement_type": "OPC",
                  "factors": None, "costs": COSTS, "robust": True,
                  "timestamp": "2026-07-01T00:00:00+00:00"}


def test_compute_metrics_defaults_bit_identical_to_pre_wp_e(predictor):
    """The WP-E gate: with transport_detail=False (default) and no exotics,
    carbon/cost/curing are unchanged from calling compute_metrics with none of
    the new kwargs at all -- the disclosure fields are new keys only."""
    exotic = _no_exotics()
    base = compute_metrics(MIX, exotic, COSTS, predictor)
    explicit = compute_metrics(MIX, exotic, COSTS, predictor,
                               transport_detail=False, site_temp_c=20.0)
    for key in ("carbon", "carbon_as_placed", "cost", "cost_as_placed", "curing"):
        assert base[key] == explicit[key], key


def test_compute_metrics_carbon_interval_brackets_displayed_carbon(predictor):
    exotic = _no_exotics()
    exotic["silica_fume"] = 50.0
    m = compute_metrics(MIX, exotic, COSTS, predictor, carbon_kwargs={"transport_km": 80.0})
    assert m["carbon_interval_lo"] <= m["carbon"] <= m["carbon_interval_hi"]
    # Non-degenerate: the registry declares nonzero uncertainty for every material
    # in this mix, so the band must have positive width.
    assert m["carbon_interval_hi"] > m["carbon_interval_lo"]


def test_compute_metrics_carbon_interval_advanced_and_lc3_also_brackets(predictor):
    """The interval is re-centered on the DISPLAYED carbon (see
    ui_logic._disclosure_metrics), so lo <= carbon <= hi holds even where the
    advanced tier's clinker chemistry diverges from the flat registry factor
    carbon_interval itself is built on."""
    exotic = _no_exotics()
    for cement_type in ("OPC", "LC3"):
        m = compute_metrics(MIX, exotic, COSTS, predictor, advanced=True,
                            carbon_kwargs={"cement_type": cement_type, "transport_km": 40.0})
        assert m["carbon_interval_lo"] <= m["carbon"] <= m["carbon_interval_hi"], cement_type


def test_compute_metrics_thermal_and_carbonation_fields(predictor):
    exotic = _no_exotics()
    m = compute_metrics(MIX, exotic, COSTS, predictor)
    assert m["delta_t_adiabatic_C"] is not None and m["delta_t_adiabatic_C"] > 0
    assert m["carbonation_uptake_bound_kg_m3"] >= 0.0
    # This mix's cement dosage is well above the ACI-207 mass-pour threshold.
    assert m["mass_pour_flag"] is not None


def test_compute_metrics_thermal_degrades_to_none_on_lc3(predictor):
    exotic = _no_exotics()
    m = compute_metrics(MIX, exotic, COSTS, predictor, advanced=True,
                        carbon_kwargs={"cement_type": "LC3"})
    assert m["delta_t_adiabatic_C"] is None
    assert m["mass_pour_flag"] is None
    assert m["carbonation_uptake_bound_kg_m3"] == 0.0


# --- Decision 1: secondary maturity-based curing (no switchover) ------------------

def test_curing_maturity_days_is_secondary_not_a_switchover(predictor):
    """`curing` (the primary heuristic) is untouched; `curing_maturity_days` is a
    separate, additional key that may legitimately disagree with it."""
    exotic = _no_exotics()
    m = compute_metrics(MIX, exotic, COSTS, predictor)
    from src.chemistry_simple import estimate_curing_time
    assert m["curing"] == estimate_curing_time(mix_dict(MIX))
    assert m["curing_maturity_days"] is not None
    assert m["curing_maturity_days"] != m["curing"]  # different quantities, expected to differ


def test_curing_maturity_days_none_on_lc3(predictor):
    exotic = _no_exotics()
    m = compute_metrics(MIX, exotic, COSTS, predictor, advanced=True,
                        carbon_kwargs={"cement_type": "LC3"})
    assert m["curing_maturity_days"] is None


def test_curing_maturity_days_decreases_with_site_temp(predictor):
    exotic = _no_exotics()
    cold = compute_metrics(MIX, exotic, COSTS, predictor, site_temp_c=5.0)
    warm = compute_metrics(MIX, exotic, COSTS, predictor, site_temp_c=35.0)
    assert cold["curing_maturity_days"] > warm["curing_maturity_days"] > 0


def test_curing_maturity_days_inert_default_site_temp_matches_explicit_20(predictor):
    exotic = _no_exotics()
    default = compute_metrics(MIX, exotic, COSTS, predictor)
    explicit = compute_metrics(MIX, exotic, COSTS, predictor, site_temp_c=20.0)
    assert default["curing_maturity_days"] == explicit["curing_maturity_days"]


# --- Decision 2: per-material transport, default-off -------------------------------

def test_carbon_for_mode_transport_detail_default_off_bit_identical():
    d = mix_dict(MIX)
    assert (carbon_for_mode(d, advanced=False, transport_km=100.0)
            == carbon_for_mode(d, advanced=False, transport_km=100.0, transport_detail=False))


def test_carbon_breakdown_transport_detail_splits_and_reconciles():
    from src.ui_logic import carbon_breakdown
    d = mix_dict(MIX)
    bd = carbon_breakdown(d, advanced=False, transport_km=100.0, transport_detail=True)
    assert "transport" not in bd
    assert "transport_registry" in bd and "transport_global" in bd
    total_via_breakdown = sum(bd.values())
    total_via_mode = carbon_for_mode(d, advanced=False, transport_km=100.0, transport_detail=True)
    assert total_via_breakdown == pytest.approx(total_via_mode)


def test_transport_detail_registry_matches_material_transport_carbon_directly():
    """`transport_registry` must equal `materials.material_transport_carbon` called
    independently on the same mix -- not just be internally self-consistent."""
    from src.ui_logic import carbon_breakdown
    from src.materials import material_transport_carbon
    d = mix_dict(MIX)  # water, superplasticizer have no registry transport block
    bd_on = carbon_breakdown(d, advanced=False, transport_km=100.0, transport_detail=True)
    assert bd_on["transport_registry"] == pytest.approx(material_transport_carbon(d))
    # water + superplasticizer (no registry block) at 100 km, 0.1 kg/t.km:
    expected_global = ((d["water"] + d["superplasticizer"]) / 1000.0) * 100.0 * 0.1
    assert bd_on["transport_global"] == pytest.approx(expected_global)


def test_compute_metrics_transport_detail_on_matches_breakdown(predictor):
    from src.ui_logic import carbon_breakdown
    exotic = _no_exotics()
    exotic["silica_fume"] = 20.0
    carbon_kwargs = {"transport_km": 120.0}
    m = compute_metrics(MIX, exotic, COSTS, predictor, carbon_kwargs=carbon_kwargs,
                        transport_detail=True)
    d = mix_dict(MIX)
    bd = carbon_breakdown(d, transport_km=120.0, exotic=exotic, transport_detail=True)
    assert sum(bd.values()) == pytest.approx(m["carbon"])


# --- D1/D2/D3/C1/C3 ticket rows -----------------------------------------------------

def test_mix_ticket_carbon_interval_rows_bracket_total(predictor):
    d = mix_dict(MIX)
    exotic = _no_exotics()
    exotic["silica_fume"] = 30.0
    m = compute_metrics(MIX, exotic, COSTS, predictor, carbon_kwargs={"transport_km": 60.0})
    config = {**DEFAULT_CONFIG, "transport_km": 60.0}
    csv = mix_ticket(d, m, config, exotic=exotic)
    lines = csv.splitlines()
    lo = float(next(line for line in lines if line.startswith("carbon_kgCO2,interval_lo,")).split(",")[2])
    hi = float(next(line for line in lines if line.startswith("carbon_kgCO2,interval_hi,")).split(",")[2])
    total = float(next(line for line in lines if line.startswith("carbon_kgCO2,TOTAL,")).split(",")[2])
    assert lo <= total <= hi


def test_mix_ticket_carbonation_row_present_and_excluded_from_total(predictor):
    d = mix_dict(MIX)
    exotic = _no_exotics()
    m = compute_metrics(MIX, exotic, COSTS, predictor)
    csv = mix_ticket(d, m, DEFAULT_CONFIG, exotic=exotic)
    lines = csv.splitlines()
    bound_line = next(line for line in lines
                      if line.startswith("carbon_kgCO2,carbonation_uptake_bound_informational,"))
    bound = float(bound_line.split(",")[2])
    assert bound >= 0.0
    total_line = next(line for line in lines if line.startswith("carbon_kgCO2,TOTAL,"))
    total = float(total_line.split(",")[2])
    # The bound is informational -- adding it to the recomputed per-source sum
    # must NOT be needed to reach TOTAL (i.e. it was never folded in).
    bd_sum = sum(
        float(line.split(",")[2]) for line in lines
        if line.startswith("carbon_kgCO2,") and line.split(",")[1]
        not in ("TOTAL", "TOTAL_as_placed", "interval_lo", "interval_hi",
                "carbonation_uptake_bound_informational")
    )
    assert bd_sum == pytest.approx(total, abs=0.05)


def test_mix_ticket_thermal_rows_present(predictor):
    d = mix_dict(MIX)
    exotic = _no_exotics()
    m = compute_metrics(MIX, exotic, COSTS, predictor)
    csv = mix_ticket(d, m, DEFAULT_CONFIG, exotic=exotic)
    assert any(line.startswith("thermal,delta_t_adiabatic_C,") for line in csv.splitlines())


def test_mix_ticket_curing_maturity_row_present_and_blank_on_lc3(predictor):
    d = mix_dict(MIX)
    exotic = _no_exotics()
    m = compute_metrics(MIX, exotic, COSTS, predictor)
    csv = mix_ticket(d, m, DEFAULT_CONFIG, exotic=exotic)
    row = next(line for line in csv.splitlines()
              if line.startswith("prediction,curing_maturity_days_uncalibrated,"))
    assert row.split(",")[2] != ""

    m_lc3 = compute_metrics(MIX, exotic, COSTS, predictor, advanced=True,
                            carbon_kwargs={"cement_type": "LC3"})
    config_lc3 = {**DEFAULT_CONFIG, "advanced": True, "cement_type": "LC3"}
    csv_lc3 = mix_ticket(d, m_lc3, config_lc3, exotic=exotic)
    row_lc3 = next(line for line in csv_lc3.splitlines()
                  if line.startswith("prediction,curing_maturity_days_uncalibrated,"))
    assert row_lc3.split(",")[2] == ""


def test_mix_ticket_allocation_and_vintage_rows(predictor):
    d = mix_dict(MIX)
    exotic = _no_exotics()
    m = compute_metrics(MIX, exotic, COSTS, predictor)
    csv = mix_ticket(d, m, DEFAULT_CONFIG, exotic=exotic)
    lines = csv.splitlines()
    assert any(line.startswith("allocation,cement,") for line in lines)
    assert any(line.startswith("vintage,cement,") for line in lines)
    assert any(line == "allocation,cement,process" for line in lines)


def test_mix_ticket_compliance_warning_row_exactly_when_restricted_material_dosed(predictor):
    d = mix_dict(MIX)
    exotic_clean = _no_exotics()
    m_clean = compute_metrics(MIX, exotic_clean, COSTS, predictor)
    csv_clean = mix_ticket(d, m_clean, DEFAULT_CONFIG, exotic=exotic_clean)
    assert not any(line.startswith("warning,") for line in csv_clean.splitlines())

    exotic_dosed = _no_exotics()
    exotic_dosed["calcium_chloride"] = 5.0
    m_dosed = compute_metrics(MIX, exotic_dosed, COSTS, predictor)
    csv_dosed = mix_ticket(d, m_dosed, DEFAULT_CONFIG, exotic=exotic_dosed)
    warning_lines = [line for line in csv_dosed.splitlines() if line.startswith("warning,calcium_chloride,")]
    assert len(warning_lines) == 1
    assert "ACI 318" in warning_lines[0]


def test_mix_ticket_transport_detail_off_no_per_material_rows(predictor):
    d = mix_dict(MIX)
    exotic = _no_exotics()
    m = compute_metrics(MIX, exotic, COSTS, predictor)
    csv = mix_ticket(d, m, DEFAULT_CONFIG, exotic=exotic)
    assert not any(line.startswith("transport_detail,") for line in csv.splitlines())
    assert not any(line.startswith("carbon_kgCO2,transport_registry,") for line in csv.splitlines())


def test_mix_ticket_transport_detail_on_discloses_per_material_mode_km(predictor):
    d = mix_dict(MIX)
    exotic = _no_exotics()
    config = {**DEFAULT_CONFIG, "transport_detail": True, "transport_km": 50.0}
    m = compute_metrics(MIX, exotic, COSTS, predictor, carbon_kwargs={"transport_km": 50.0},
                        transport_detail=True)
    csv = mix_ticket(d, m, config, exotic=exotic)
    lines = csv.splitlines()
    assert any(line.startswith("transport_detail,cement,") for line in lines)
    assert any(line.startswith("carbon_kgCO2,transport_registry,") for line in lines)
    assert any(line.startswith("carbon_kgCO2,transport_global,") for line in lines)
    assert any(line == "config,transport_detail,True" for line in lines)
    # Reconciliation still holds end to end under transport_detail.
    total = float(next(line for line in lines if line.startswith("carbon_kgCO2,TOTAL,")).split(",")[2])
    assert total == pytest.approx(m["carbon"], abs=0.05)


def test_mix_ticket_site_temp_c_config_row(predictor):
    d = mix_dict(MIX)
    exotic = _no_exotics()
    m = compute_metrics(MIX, exotic, COSTS, predictor, site_temp_c=25.0)
    config = {**DEFAULT_CONFIG, "site_temp_c": 25.0}
    csv = mix_ticket(d, m, config, exotic=exotic)
    assert "config,site_temp_c,25.0" in csv.splitlines()


def test_mix_ticket_disclosure_falls_back_for_metrics_missing_new_fields(predictor):
    """recommend_recipe-style tickets (whose metrics dict predates WP-E) must
    still get every disclosure row -- recomputed fresh from mix/config."""
    d = mix_dict(MIX)
    m = compute_metrics(MIX, _no_exotics(), COSTS, predictor)
    legacy_metrics = {k: v for k, v in m.items()
                      if k not in ("carbon_interval_lo", "carbon_interval_hi",
                                   "delta_t_adiabatic_C", "mass_pour_flag",
                                   "carbonation_uptake_bound_kg_m3", "curing_maturity_days")}
    csv = mix_ticket(d, legacy_metrics, DEFAULT_CONFIG)
    lines = csv.splitlines()
    assert any(line.startswith("carbon_kgCO2,interval_lo,") for line in lines)
    assert any(line.startswith("thermal,delta_t_adiabatic_C,") for line in lines)
    assert any(line.startswith("prediction,curing_maturity_days_uncalibrated,") for line in lines)
