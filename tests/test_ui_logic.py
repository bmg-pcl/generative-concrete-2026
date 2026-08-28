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
    carbon_term,
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
    slump_caveat,
    compliance_advisory_text,
    compliance_matrix,
)
from src.chemistry_simple import calculate_embodied_carbon, calculate_mix_cost, CARBON_FACTORS
from src.chemistry_advanced import embodied_carbon_advanced
from src.exotics import exotic_strength_delta
from src.models import StrengthPredictor
from src.properties import slump_estimate
from src.compliance import set_packs_path

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


# --- R8.5 P1: the coherence contract, kept forever ---------------------------------
#
# ui_logic's module docstring promises "ONE carbon path, ONE metrics path, ONE
# fitness path" -- the audit for docs/specs/R8.5 found exactly one live
# divergence from that promise: `ui/config.py:161` builds `ctx.carbon_kwargs`
# WITHOUT `transport_detail`, so with the per-material transport toggle ON the
# ticket (compute_metrics) shows per-material transport carbon while the
# optimizer (scalarized_fitness/recommend_recipe/run_nsga) still minimizes the
# global-km path. `ui/config.py` is WP-2's file (P4's package, dispatched after
# this one) -- the one-line dict-entry fix lands there. This test suite is the
# ENGINE-side half of the contract: it proves that once `transport_detail`
# arrives inside `carbon_kwargs` (exactly the shape the WP-2 fix produces),
# every engine consumer honours it identically to `compute_metrics`, regardless
# of when the UI-side dict entry lands. It is written so it ALREADY exercises
# that exact shape (a `carbon_kwargs` dict carrying a `transport_detail` key)
# rather than waiting for WP-2 -- which is how it caught a second, more subtle
# defect: `compute_metrics` used to ALSO accept `transport_detail` as its own
# keyword parameter and unconditionally re-passed it alongside `**carbon_kwargs`
# to `carbon_for_mode`, so a `carbon_kwargs` dict carrying that key (the post-fix
# shape) would raise "got multiple values for keyword argument 'transport_detail'"
# on the very first call -- fixed in `compute_metrics` by letting `carbon_kwargs`
# win when both are present (see its docstring).

def _factors_variants():
    """None (registry defaults) and a real override -- every core key present,
    so the override changes the arithmetic without silently zeroing materials
    `factors` doesn't mention (carbon_for_mode's `factors` dict REPLACES, not
    merges with, the registry defaults -- see chemistry_simple.calculate_embodied_carbon)."""
    overridden = dict(CARBON_FACTORS)
    overridden["cement"] = overridden["cement"] * 0.5
    return (None, overridden)


def _clinker_source_variants():
    """None (legacy default) and a real descriptor. Only affects the advanced
    tier (carbon_for_mode's own contract) -- included in every combination
    anyway so the coherence test also proves it stays a true no-op in simple
    mode on BOTH sides of the comparison, not just tested in isolation."""
    return (None, {"kiln_fuel": "natural_gas", "electricity": "hydro"})


_P1_AXES = [
    (advanced, transport_km, transport_detail, factors_override, clinker_source)
    for advanced in (False, True)
    for transport_km in (0.0, 500.0)
    for transport_detail in (False, True)
    for factors_override in _factors_variants()
    for clinker_source in _clinker_source_variants()
]


@pytest.mark.parametrize(
    "advanced,transport_km,transport_detail,factors_override,clinker_source", _P1_AXES,
    ids=[f"advanced={a}-km={k}-detail={t}-factors={'override' if f else 'default'}-"
         f"clinker={'set' if c else 'none'}"
         for a, k, t, f, c in _P1_AXES],
)
def test_p1_scalarized_fitness_carbon_term_matches_compute_metrics(
    predictor, advanced, transport_km, transport_detail, factors_override, clinker_source,
):
    """The durable gate this spec names explicitly: `scalarized_fitness`'s carbon
    term `==` `compute_metrics(...)["carbon"]` for the same mix, across every
    axis of the R8 config surface -- simple/advanced x transport_km {0, 500} x
    transport_detail {off, on} x factor overrides x clinker_source. `w_strength`
    and `w_cost` are zeroed so `fitness == -carbon` isolates the carbon term
    exactly; `exotic={}` on the compute_metrics side matches scalarized_fitness
    having no exotic dosing concept at all (both `exotic_carbon({})` and
    `carbon_for_mode(..., exotic=None)` contribute zero -- see
    test_carbon_for_mode_exotic_default_is_bit_identical)."""
    carbon_kwargs = {
        "transport_km": transport_km,
        "cement_type": "OPC",
        "factors": factors_override,
        "clinker_source": clinker_source,
        "transport_detail": transport_detail,
    }
    fitness = scalarized_fitness(MIX, COSTS, predictor, 0.0, 1.0, 0.0,
                                 advanced=advanced, carbon_kwargs=carbon_kwargs)
    fitness_carbon = -fitness

    m = compute_metrics(MIX, {}, COSTS, predictor, advanced=advanced,
                        carbon_kwargs=carbon_kwargs)

    assert fitness_carbon == pytest.approx(m["carbon"]), (
        f"scalarized_fitness carbon term diverged from compute_metrics carbon: "
        f"advanced={advanced} transport_km={transport_km} "
        f"transport_detail={transport_detail} clinker_source={clinker_source}"
    )


def test_p1_compute_metrics_carbon_kwargs_transport_detail_overrides_explicit_param():
    """The collision-avoidance regression: `compute_metrics` must not raise when
    `carbon_kwargs` carries `transport_detail` (the post-WP-2 shape) -- and once
    it does, that value governs (matches calling with the EQUIVALENT explicit
    parameter, never both silently disagreeing)."""
    predictor = StrengthPredictor()
    exotic = {}
    via_dict = compute_metrics(MIX, exotic, COSTS, predictor,
                               carbon_kwargs={"transport_km": 300.0, "transport_detail": True})
    via_param = compute_metrics(MIX, exotic, COSTS, predictor,
                                carbon_kwargs={"transport_km": 300.0},
                                transport_detail=True)
    assert via_dict["carbon"] == pytest.approx(via_param["carbon"])
    # And the explicit parameter is silently ignored (not summed/double-applied)
    # when carbon_kwargs disagrees with it -- carbon_kwargs is the one source.
    dict_says_off_param_says_on = compute_metrics(
        MIX, exotic, COSTS, predictor,
        carbon_kwargs={"transport_km": 300.0, "transport_detail": False},
        transport_detail=True,
    )
    off_explicit = compute_metrics(MIX, exotic, COSTS, predictor,
                                   carbon_kwargs={"transport_km": 300.0},
                                   transport_detail=False)
    assert dict_says_off_param_says_on["carbon"] == pytest.approx(off_explicit["carbon"])


def test_p1_recommend_recipe_carbon_matches_compute_metrics_for_returned_mix():
    """`recommend_recipe`'s returned `"carbon"` must equal `compute_metrics`'s
    carbon for the SAME returned mix under the SAME carbon_kwargs, with
    transport_detail on and off -- the third leg of the coherence contract (the
    NSGA leg lives in tests/test_nsga.py)."""
    from src.bayesian import BayesFlowExplorer
    np.random.seed(0)
    explorer = BayesFlowExplorer()
    for transport_detail in (False, True):
        carbon_kwargs = {"transport_km": 200.0, "transport_detail": transport_detail}
        rec = recommend_recipe(explorer, 40.0, method="ga", costs=COSTS,
                               carbon_kwargs=carbon_kwargs)
        m = compute_metrics(rec["mix"], {}, COSTS, explorer.predictor,
                            carbon_kwargs=carbon_kwargs)
        assert rec["carbon"] == pytest.approx(m["carbon"]), transport_detail


# --- R8.5 P2: robust carbon -- optimize the upper bound, symmetric with robust strength --
#
# R1's flagship move was optimizing the strength LOWER bound; carbon_term's
# robust_carbon mirrors it on the carbon side by optimizing the UPPER bound
# (materials.carbon_interval's +1.96*sigma, re-centered on the point total --
# same convention _disclosure_metrics's carbon_interval_hi already established).

def test_p2_carbon_term_point_mode_matches_carbon_for_mode():
    """robust_carbon=False (the default) is exactly carbon_for_mode -- no new
    arithmetic on the default path."""
    d = mix_dict(MIX)
    carbon_kwargs = {"transport_km": 150.0}
    assert carbon_term(d, advanced=False, carbon_kwargs=carbon_kwargs) == (
        carbon_for_mode(d, advanced=False, **carbon_kwargs)
    )
    assert carbon_term(d, advanced=False, carbon_kwargs=carbon_kwargs, robust_carbon=False) == (
        carbon_for_mode(d, advanced=False, **carbon_kwargs)
    )


def test_p2_scalarized_fitness_robust_carbon_default_bit_identical(predictor):
    """No caller touches `robust_carbon` (default False) -> identical to before
    the flag existed."""
    carbon_kwargs = {"transport_km": 200.0}
    omitted = scalarized_fitness(MIX, COSTS, predictor, 1.0, 0.05, 0.5,
                                 carbon_kwargs=carbon_kwargs)
    explicit_false = scalarized_fitness(MIX, COSTS, predictor, 1.0, 0.05, 0.5,
                                        carbon_kwargs=carbon_kwargs, robust_carbon=False)
    assert omitted == explicit_false


def test_p2_robust_carbon_zero_uncertainty_bit_identical(predictor, monkeypatch):
    """Gate: with every registry uncertainty patched to zero, robust_carbon's
    output is bit-identical to point mode (sigma == 0 collapses the upper bound
    onto the point total exactly)."""
    import src.ui_logic as ui_logic
    monkeypatch.setattr(ui_logic, "factor_uncertainties_view", lambda: {})
    carbon_kwargs = {"transport_km": 150.0}
    point = scalarized_fitness(MIX, COSTS, predictor, 0.0, 1.0, 0.0,
                               carbon_kwargs=carbon_kwargs, robust_carbon=False)
    robust = scalarized_fitness(MIX, COSTS, predictor, 0.0, 1.0, 0.0,
                                carbon_kwargs=carbon_kwargs, robust_carbon=True)
    assert robust == point


def test_p2_robust_carbon_prefers_better_characterized_composition():
    """Gate: two mixes with EQUAL point carbon but different composition
    uncertainty -- robust_carbon must rank the mix built from the LOWER-
    relative-uncertainty material as cheaper, even though their plain point
    carbon ties exactly. Uses the spec's own example: fly ash's factor is tiny
    but +/-50% uncertain; slag's is larger but only +/-30%."""
    from src.materials import carbon_factors_view, factor_uncertainties_view
    factors = carbon_factors_view()
    uncertainties = factor_uncertainties_view()
    assert uncertainties["slag"] < uncertainties["ash"]  # 0.30 vs 0.50

    # A shared baseline (cement/water/etc, from MIX); swap in slag vs ash masses
    # chosen so each contributes IDENTICAL point carbon.
    contribution = 20.0  # kg CO2/m3, arbitrary but shared
    slag_qty = contribution / factors["slag"]
    ash_qty = contribution / factors["ash"]
    base = dict(zip(PARAM_NAMES, MIX))
    mix_slag = mix_dict([{**base, "slag": slag_qty, "ash": 0.0}[p] for p in PARAM_NAMES])
    mix_ash = mix_dict([{**base, "slag": 0.0, "ash": ash_qty}[p] for p in PARAM_NAMES])

    point_slag = carbon_for_mode(mix_slag, advanced=False)
    point_ash = carbon_for_mode(mix_ash, advanced=False)
    assert point_slag == pytest.approx(point_ash)  # equal point carbon, by construction

    robust_slag = carbon_term(mix_slag, advanced=False, robust_carbon=True)
    robust_ash = carbon_term(mix_ash, advanced=False, robust_carbon=True)
    assert robust_slag < robust_ash  # slag (better-characterized) wins under robust_carbon


def test_p2_robust_carbon_monotone_in_tightened_uncertainty(monkeypatch):
    """Gate: tightening ONE material's uncertainty (an EPD attaching a tighter
    number, per the spec's 'attach an EPD, your guaranteed number improves'
    framing) lowers that mix's robust_carbon objective MONOTONICALLY."""
    import src.ui_logic as ui_logic
    from src.materials import factor_uncertainties_view as real_view
    base_unc = real_view()
    d = mix_dict(MIX)  # MIX carries slag=100 -- a nonzero mass so tightening moves sigma

    def patched(u):
        merged = dict(base_unc)
        merged["slag"] = u
        return merged

    values = []
    for u in (0.30, 0.20, 0.10, 0.0):
        monkeypatch.setattr(ui_logic, "factor_uncertainties_view", lambda u=u: patched(u))
        values.append(carbon_term(d, advanced=False, robust_carbon=True))
    assert all(values[i] > values[i + 1] for i in range(len(values) - 1)), values


def test_p2_recommend_recipe_discloses_carbon_basis():
    """Gate: the returned dict always carries `carbon_basis`, correctly set."""
    from src.bayesian import BayesFlowExplorer
    np.random.seed(0)
    explorer = BayesFlowExplorer()
    rec_point = recommend_recipe(explorer, 40.0, method="ga", costs=COSTS)
    assert rec_point["carbon_basis"] == "point"
    rec_explicit_point = recommend_recipe(explorer, 40.0, method="ga", costs=COSTS,
                                          robust_carbon=False)
    assert rec_explicit_point["carbon_basis"] == "point"
    rec_robust = recommend_recipe(explorer, 40.0, method="ga", costs=COSTS, robust_carbon=True)
    assert rec_robust["carbon_basis"] == "upper_95"


def test_p2_recommend_recipe_robust_carbon_default_bit_identical():
    """No caller touches `robust_carbon` -> identical mix chosen and identical
    carbon reported (same GA seed)."""
    from src.bayesian import BayesFlowExplorer
    np.random.seed(0)
    explorer = BayesFlowExplorer()
    omitted = recommend_recipe(explorer, 40.0, method="ga", costs=COSTS)
    np.random.seed(0)
    explorer2 = BayesFlowExplorer()
    explicit_false = recommend_recipe(explorer2, 40.0, method="ga", costs=COSTS,
                                      robust_carbon=False)
    assert np.array_equal(omitted["mix"], explicit_false["mix"])
    assert omitted["carbon"] == explicit_false["carbon"]


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


# ===================================================================================
# R8.1 WP-3 (Wave B): slump display -- compute_metrics + mix_ticket integration.
# The two non-negotiable display rules (see docs/specs/R8.1's WP-1b section): show
# the POINT ESTIMATE, state the interval width as a plain caveat, and NEVER render
# it as a bound/guarantee. Out-of-support mixes show basis="heuristic", never a
# bare model number.
# ===================================================================================

# A row drawn straight from the committed slump corpus (data/slump_test.data row
# 10) -- genuinely IN-SUPPORT for the slump model, unlike MIX above (which is
# in-support for STRENGTH but sits outside the slump corpus's envelope -- see
# tests/test_properties.py::test_strength_in_support_slump_out_of_support, the
# same per-property-gate finding R8.1 WP-1 exhibited).
IN_SUPPORT_SLUMP_MIX = [145, 106, 136, 208, 10, 751, 883, 28]


def test_mix_out_of_slump_support_is_in_strength_support(predictor):
    """Exhibits the per-property-gate finding this integration relies on: the
    everyday MIX used throughout this file is comfortably in-support for
    STRENGTH but outside the slump model's OWN envelope (every slump-corpus row
    used SP >= 4.4 kg/m3; MIX's SP=5 is fine on that axis, but the full 7-D kNN
    distance still lands it outside -- see docs/specs/R8.1)."""
    m = compute_metrics(MIX, _no_exotics(), COSTS, predictor)
    assert m["in_support"] is True          # strength: in-support
    assert m["slump_basis"] == "heuristic"  # slump: NOT in-support -- its own gate


def test_compute_metrics_slump_model_basis_when_in_support(predictor):
    m = compute_metrics(IN_SUPPORT_SLUMP_MIX, _no_exotics(), COSTS, predictor)
    assert m["slump_basis"] == "model"
    assert m["slump_in_support"] is True
    assert m["slump_cm"] is not None
    # WP-1's coherence gate, reused at the integration layer: lo <= point <= hi.
    assert m["slump_lo"] <= m["slump_cm"] <= m["slump_hi"]
    assert m["slump_reason"] is None
    # Cross-check against calling slump_estimate directly on the same mix.
    direct = slump_estimate(mix_dict(IN_SUPPORT_SLUMP_MIX))
    assert m["slump_cm"] == pytest.approx(direct["slump_cm"])


def test_compute_metrics_slump_heuristic_basis_when_out_of_support(predictor):
    m = compute_metrics(MIX, _no_exotics(), COSTS, predictor)
    assert m["slump_basis"] == "heuristic"
    assert m["slump_in_support"] is False
    # Never a confident model number outside the trained envelope.
    assert m["slump_cm"] is None
    assert m["slump_lo"] is None and m["slump_hi"] is None
    assert m["slump_reason"] is not None


def test_slump_caveat_never_phrases_the_interval_as_a_bound():
    text = slump_caveat(10.0, 22.0)
    assert "NOT a guaranteed bound" in text
    assert "guarantee" not in text.lower().replace("not a guaranteed", "")
    # Width and half-width are both stated plainly.
    assert "12.0 cm" in text   # interval width (22 - 10)
    assert "±6.0 cm" in text   # half-width, the "at 90%" figure
    # The heuristic (no-interval) path states plainly that there is nothing to bound.
    assert "No measured interval" in slump_caveat(None, None)


def test_mix_ticket_slump_rows_model_basis(predictor):
    d = mix_dict(IN_SUPPORT_SLUMP_MIX)
    m = compute_metrics(IN_SUPPORT_SLUMP_MIX, _no_exotics(), COSTS, predictor)
    csv = mix_ticket(d, m, DEFAULT_CONFIG)
    lines = csv.splitlines()
    cm_row = next(line for line in lines if line.startswith("prediction,slump_cm_model,"))
    assert float(cm_row.split(",")[2]) == pytest.approx(m["slump_cm"], abs=0.05)
    width_row = next(line for line in lines if line.startswith("prediction,slump_interval_width_cm,"))
    assert float(width_row.split(",")[2]) == pytest.approx(m["slump_hi"] - m["slump_lo"], abs=0.05)
    note_row = next(line for line in lines if line.startswith("note,slump_interval,"))
    assert "NOT a guaranteed bound" in note_row
    # Never rendered as an interval90-style bound row (that phrasing is reserved
    # for the strength interval, which genuinely is conformalised as a bound).
    assert not any("slump_interval90" in line for line in lines)


def test_mix_ticket_slump_rows_heuristic_basis(predictor):
    d = mix_dict(MIX)
    m = compute_metrics(MIX, _no_exotics(), COSTS, predictor)
    csv = mix_ticket(d, m, DEFAULT_CONFIG)
    lines = csv.splitlines()
    cm_row = next(line for line in lines if line.startswith("prediction,slump_cm_heuristic,"))
    assert "n/a" in cm_row
    # No numeric model interval width row on the heuristic path.
    assert not any(line.startswith("prediction,slump_interval_width_cm,") for line in lines)
    note_row = next(line for line in lines if line.startswith("note,slump_interval,"))
    assert "heuristic fallback" in note_row.lower()


def test_mix_ticket_slump_fallback_for_metrics_missing_new_fields(predictor):
    """A metrics dict predating this wave (e.g. recommend_recipe's own dict, which
    carries no slump_* keys at all) must still get every slump row -- recomputed
    fresh via slump_estimate(mix), same fallback shape as DISCLOSURE_KEYS."""
    d = mix_dict(IN_SUPPORT_SLUMP_MIX)
    m = compute_metrics(IN_SUPPORT_SLUMP_MIX, _no_exotics(), COSTS, predictor)
    legacy_metrics = {k: v for k, v in m.items() if not k.startswith("slump_")}
    csv = mix_ticket(d, legacy_metrics, DEFAULT_CONFIG)
    lines = csv.splitlines()
    assert any(line.startswith("prediction,slump_cm_model,") for line in lines)
    assert any(line.startswith("prediction,slump_interval_width_cm,") for line in lines)
    assert any(line.startswith("note,slump_interval,") for line in lines)


# ===================================================================================
# R8.2 WP-3 (Wave B): compliance integration -- compute_metrics + mix_ticket +
# compliance_matrix (the cross-jurisdiction headline). Defaults must be INERT
# (bit-identical to every pre-existing number); UNKNOWN must never upgrade to
# PASS; the advisory row is mandatory whenever a verdict exists.
# ===================================================================================

# A small, fully-specified pack (every rule present on both classes) so a plain
# PASS/FAIL is actually reachable -- unlike the shipped real packs, which (per
# WP-2's own honest "omit rather than guess" discipline) omit max_scm_fraction
# on EVERY class, so a real-pack verdict is always at least UNKNOWN. Mirrors
# compliance.py's own `_fixture` pack in spirit (full-coverage + gap classes),
# built inline here so this integration layer's tests do not depend on the
# real packs' completeness (which is honestly out of WP-3's control).
_TEST_PACK = {
    "pack_id": "testpack", "name": "WP-3 integration test pack",
    "jurisdiction": "N/A (src/ui_logic.py test fixture)",
    "source": {"standard": "TEST-STD-1", "table": "T.1", "verified": False,
              "verification_note": "Synthetic, not a real standard."},
    "strength_basis": "conformal_lower_bound",
    "classes": {
        "T1": {"max_w_b": 0.50, "min_cement_kg_m3": 300, "min_strength_MPa": 30,
              "min_air_pct": None, "max_scm_fraction": {"ash": 0.33, "slag": 0.80}},
        "T2": {"max_w_b": 0.30, "min_cement_kg_m3": 500, "min_strength_MPa": 60,
              "min_air_pct": None, "max_scm_fraction": {"ash": 0.33, "slag": 0.80}},
        "GAP": {"max_w_b": 0.50},   # every other rule absent -> always UNKNOWN
    },
}


@pytest.fixture()
def test_pack_dir(tmp_path):
    """Drop `_TEST_PACK` into a temp registry dir and point compliance.py at it
    for the duration of one test -- the same `set_packs_path` pluggability WP-1
    gates on ("a jurisdiction is a JSON drop-in, not a code change"). Restores the
    default registry afterward so this file cannot poison other test modules."""
    import json
    (tmp_path / "testpack.json").write_text(json.dumps(_TEST_PACK))
    set_packs_path(str(tmp_path))
    try:
        yield str(tmp_path)
    finally:
        set_packs_path(None)


def test_compute_metrics_compliance_inert_by_default(predictor):
    """The Wave B gate: no pack selected -> every pre-existing number/key is
    bit-identical to calling compute_metrics with none of the new kwargs, and
    the new `compliance` key is exactly None (not an empty dict, not omitted)."""
    exotic = _no_exotics()
    implicit = compute_metrics(MIX, exotic, COSTS, predictor)
    explicit = compute_metrics(MIX, exotic, COSTS, predictor,
                               exposure_pack=None, exposure_class=None, air_pct=None)
    assert implicit["compliance"] is None
    assert explicit["compliance"] is None
    pre_existing_keys = [k for k in implicit if k not in
                         ("compliance",) and not k.startswith("slump_")]
    for k in pre_existing_keys:
        assert implicit[k] == explicit[k], k


def test_compute_metrics_compliance_unresolvable_ids_are_inert(predictor):
    """An exposure_pack/exposure_class that does not resolve to a real pack/class
    degrades to inert (None), never raises -- this is an advisory UI feature, not
    a validated boundary (the CLI's --exposure flag is the validated boundary)."""
    exotic = _no_exotics()
    m1 = compute_metrics(MIX, exotic, COSTS, predictor, exposure_pack="does_not_exist",
                         exposure_class="XC4")
    assert m1["compliance"] is None
    m2 = compute_metrics(MIX, exotic, COSTS, predictor, exposure_pack="en206",
                         exposure_class="ZZ9")
    assert m2["compliance"] is None
    m3 = compute_metrics(MIX, exotic, COSTS, predictor, exposure_pack="en206",
                         exposure_class=None)
    assert m3["compliance"] is None


def test_compute_metrics_compliance_pass_at_exact_limit(predictor, test_pack_dir):
    """The Wave B gate: a mix exactly AT a limit PASSES (deemed-to-satisfy limits
    are inclusive)."""
    exotic = _no_exotics()
    # w/b = 150/300 = 0.50 exactly (T1's max_w_b); cement=300 exactly (T1's
    # min_cement_kg_m3); strength lower bound comfortably clears 30 MPa for a
    # 300 kg/m3 OPC mix at this w/b.
    mix = [300, 0, 0, 150, 5, 1000, 750, 28]
    m = compute_metrics(mix, exotic, COSTS, predictor,
                        exposure_pack="testpack", exposure_class="T1")
    c = m["compliance"]
    assert c is not None
    assert c["pack_id"] == "testpack" and c["class"] == "T1"
    w_b_rule = next(r for r in c["rules"] if r["rule"] == "max_w_b")
    assert w_b_rule["actual"] == pytest.approx(0.50)
    assert w_b_rule["result"] == "PASS"
    cement_rule = next(r for r in c["rules"] if r["rule"] == "min_cement_kg_m3")
    assert cement_rule["actual"] == pytest.approx(300.0)
    assert cement_rule["result"] == "PASS"
    assert c["verdict"] == "PASS"


def test_compute_metrics_compliance_fails_a_weak_mix(predictor, test_pack_dir):
    exotic = _no_exotics()
    weak_mix = [150, 0, 0, 180, 0, 1000, 750, 3]  # low cement, high w/b, age 3d
    m = compute_metrics(weak_mix, exotic, COSTS, predictor,
                        exposure_pack="testpack", exposure_class="T2")
    assert m["compliance"]["verdict"] == "FAIL"


def test_compute_metrics_compliance_unknown_rule_never_upgrades_to_pass(predictor, test_pack_dir):
    """T2's GAP-adjacent class 'GAP' only declares max_w_b -- every other rule is
    ABSENT (not null), so the verdict must be UNKNOWN even for an otherwise-
    excellent mix, never silently PASS."""
    exotic = _no_exotics()
    strong_mix = [500, 0, 0, 140, 10, 1000, 750, 90]
    m = compute_metrics(strong_mix, exotic, COSTS, predictor,
                        exposure_pack="testpack", exposure_class="GAP")
    c = m["compliance"]
    assert c["verdict"] == "UNKNOWN"
    assert c["unknown_count"] >= 1


def test_compute_metrics_compliance_uses_conformal_lower_bound_not_mean(predictor, test_pack_dir):
    """R8.2's central design decision: strength is checked against interval_lo
    (the conformal LOWER bound, already inclusive of any exotic delta), never the
    point-estimate mean."""
    exotic = _no_exotics()
    m = compute_metrics(MIX, exotic, COSTS, predictor,
                        exposure_pack="testpack", exposure_class="T1")
    rule = next(r for r in m["compliance"]["rules"] if r["rule"] == "min_strength_MPa")
    assert rule["actual"] == pytest.approx(m["interval_lo"])
    assert rule["actual"] < m["strength"]   # the lower bound is strictly below the mean
    assert "lower_bound" in rule["basis"]


def test_mix_ticket_compliance_rows_absent_when_inert(predictor):
    d = mix_dict(MIX)
    m = compute_metrics(MIX, _no_exotics(), COSTS, predictor)
    csv = mix_ticket(d, m, DEFAULT_CONFIG)
    assert not any(line.startswith("compliance,") for line in csv.splitlines())


def test_mix_ticket_compliance_rows_and_mandatory_advisory(predictor, test_pack_dir):
    d = mix_dict([300, 0, 0, 150, 5, 1000, 750, 28])
    m = compute_metrics([300, 0, 0, 150, 5, 1000, 750, 28], _no_exotics(), COSTS, predictor,
                        exposure_pack="testpack", exposure_class="T1")
    csv = mix_ticket(d, m, DEFAULT_CONFIG)
    lines = csv.splitlines()
    verdict_row = next(line for line in lines if line.startswith("compliance,testpack.T1,"))
    assert verdict_row.endswith(",PASS")
    advisory_row = next(line for line in lines if line.startswith("compliance,advisory,"))
    assert "TEST-STD-1" in advisory_row          # names the standard
    assert "NOT a certification" in advisory_row


def test_mix_ticket_compliance_unknown_rows_present_per_failing_rule(predictor, test_pack_dir):
    d = mix_dict([500, 0, 0, 140, 10, 1000, 750, 90])
    m = compute_metrics([500, 0, 0, 140, 10, 1000, 750, 90], _no_exotics(), COSTS, predictor,
                        exposure_pack="testpack", exposure_class="GAP")
    csv = mix_ticket(d, m, DEFAULT_CONFIG)
    lines = csv.splitlines()
    assert any(line.startswith("compliance,testpack.GAP,UNKNOWN") for line in lines)
    # One row per UNKNOWN/FAIL rule -- min_cement_kg_m3, min_strength_MPa,
    # min_air_pct, max_scm_fraction are all absent from GAP.
    unknown_rule_rows = [line for line in lines if line.startswith("compliance,rule_")]
    assert len(unknown_rule_rows) == 4
    assert any("UNKNOWN" in line for line in unknown_rule_rows)


def test_compliance_advisory_text_names_the_standard():
    source = {"standard": "EN 206:2013+A2:2021", "verification_note": "check it"}
    text = compliance_advisory_text(source)
    assert "EN 206:2013+A2:2021" in text
    assert "NOT a certification" in text
    assert "check it" in text


def test_compliance_matrix_shows_pass_and_fail_across_jurisdictions():
    """The R8.2 WP-3 headline gate: the cross-jurisdiction table shows at least
    one mix passing in one jurisdiction and failing/unknown in another, via
    compliance_matrix (built on compliance.compare_jurisdictions)."""
    mix = {"cement": 300, "slag": 0, "ash": 0, "water": 150, "superplasticizer": 5,
          "coarse_agg": 1000, "fine_agg": 750}
    packs = {"testpack": _TEST_PACK}
    rows = compliance_matrix(mix, strength_lo=35.0, highlight_pack="testpack",
                             highlight_class="T1", packs=packs)
    verdicts = {r["class"]: r["verdict"] for r in rows}
    assert verdicts["T1"] == "PASS"
    # T1's row is here because highlight_pack/highlight_class asked for it; add a
    # second, independent pack with a class this same mix FAILS to exhibit real
    # cross-jurisdiction variation.
    strict_pack = {
        **_TEST_PACK, "pack_id": "strictpack",
        "classes": {"S1": {"max_w_b": 0.30, "min_cement_kg_m3": 500,
                          "min_strength_MPa": 60, "min_air_pct": None,
                          "max_scm_fraction": {"ash": 0.33, "slag": 0.80}}},
    }
    packs2 = {"testpack": _TEST_PACK, "strictpack": strict_pack}
    rows2 = compliance_matrix(mix, strength_lo=35.0, highlight_pack="testpack",
                              highlight_class="T1", packs=packs2)
    verdicts2 = {(r["pack_id"], r["class"]): r["verdict"] for r in rows2}
    assert verdicts2[("testpack", "T1")] == "PASS"
    assert verdicts2[("strictpack", "S1")] == "FAIL"


def test_compliance_matrix_default_representative_class_per_pack():
    """Without a highlight, each pack contributes the class that STATES the most
    rules -- deterministic, and never a hardcoded jurisdiction list (built from
    whatever `packs` names).

    This deliberately replaces an earlier alphabetical rule. Alphabetical picked
    ACI 318's real "C0" (concrete dry or protected from moisture -- the
    not-exposed category, every rule null by construction), so the shipped table
    paired EN 206's XA1, a genuine chemical-attack requirement, against a class
    that imposes nothing to fail. It rendered as "en206: UNKNOWN / aci318: PASS"
    and invited precisely the wrong reading. Here the same degenerate case is
    "GAP" (every rule but one absent); the rule must not choose it."""
    mix = {"cement": 300, "slag": 0, "ash": 0, "water": 150, "superplasticizer": 5,
          "coarse_agg": 1000, "fine_agg": 750}
    rows = compliance_matrix(mix, strength_lo=35.0, packs={"testpack": _TEST_PACK})
    assert len(rows) == 1
    assert rows[0]["class"] != "GAP", "must not represent a pack by a rule-less class"
    assert rows[0]["class"] in ("T1", "T2")   # both state 4 rules; ties break alphabetically
    assert rows[0]["class"] == "T1"
