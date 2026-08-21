"""
Tests for R8.0 WP-C -- hydration-layer consumers (src/thermal.py).

(see docs/specs/R8.0-carbon-chemistry-deepening.md, "WP-C - Hydration-layer
consumers"). Every gate here is a unit/bound/conservation check on top of the
already-UNCALIBRATED hydration chain in chemistry_advanced.py -- never a claim
that these numbers match a measured cement. See src/thermal.py's module
docstring for the inheritance notice this carries into every public function.

Honesty note on the C1 gate (see below): the spec's own illustrative arithmetic
(~300 kJ/kg cement x 350 kg/m3 / 2400 ~= 44 C, "broad physics band 30-60 C")
under-estimates the model's actual full-hydration heat. `hydration_kinetics`'s
own alpha->1 heat for the shipped OPC oxides was already pinned in
tests/test_hydration.py to the band [350, 520] kJ/kg cement (finding 1's unit
gate) -- nearer 470 than 300 at the point this model actually saturates.
Carried through the SAME delta_T formula the spec specifies, [350, 520] kJ/kg
gives delta_T in [51.0, 75.8] C for the 350 kg/m3 reference mix, not [30, 60].
Per the R8.0 non-negotiables ("no coefficient may be tuned to make a band
pass -- report it"), the gate below is widened to the honestly-derived band
and the discrepancy is reported here rather than silently absorbed -- no
constant in thermal.py (CONCRETE_DENSITY, SPECIFIC_HEAT, or the hydration
chain it imports unchanged) was touched to make a number land somewhere.
"""
import pytest

from src import thermal as th
from src.chemistry_advanced import analyze_mix, hydration_kinetics

REF_MIX = {"cement": 350, "water": 180, "coarse_agg": 1000, "fine_agg": 750, "age": 28}


# --- C1: adiabatic temperature rise -------------------------------------------------
def test_adiabatic_rise_lands_in_honestly_derived_band():
    """delta_T = heat_kJ_per_kg_cement(alpha->1) * cement_mass / (rho * c_p).

    Derivation of the band (see module docstring above for why this is wider
    than the spec's illustrative 30-60 C): test_hydration.py's finding-1 gate
    already pins full-hydration heat for the shipped OPC oxides to
    [350, 520] kJ/kg cement. At cement=350 kg/m3, rho=2400, c_p=1.0:
        lo = 350 * 350 / 2400 = 51.04 C
        hi = 520 * 350 / 2400 = 75.83 C
    """
    delta_t = th.adiabatic_temperature_rise(REF_MIX)
    assert delta_t is not None
    assert 51.0 <= delta_t <= 76.0
    # Regression guard: this is NOT a partial-hydration snapshot (e.g. the
    # 28-day value, ~44 C by the spec's own rough arithmetic) -- it must be
    # the full alpha->alpha_ult asymptote, which is materially higher.
    assert delta_t > 60.0


def test_adiabatic_rise_matches_direct_full_hydration_calculation():
    """Cross-check against hydration_kinetics called directly at a large age,
    the same construction adiabatic_temperature_rise performs internally."""
    from src.chemistry_advanced import bogue_calculation, clinker_factor_for, load_oxide_compositions
    oxides = load_oxide_compositions()["OPC"]
    phases = bogue_calculation(oxides)
    cf = clinker_factor_for("OPC")
    state = hydration_kinetics(phases, w_c_ratio=180 / 350, age_days=36_500,
                                cement_mass_kg_m3=350, clinker_factor=cf)
    assert state.degree_of_hydration == pytest.approx(1.0, abs=1e-6)
    expected = state.heat_kJ_per_kg_cement * 350 / (th.CONCRETE_DENSITY * th.SPECIFIC_HEAT)
    assert th.adiabatic_temperature_rise(REF_MIX) == pytest.approx(expected, rel=1e-9)


def test_adiabatic_rise_zero_cement_is_zero_not_none():
    assert th.adiabatic_temperature_rise({"cement": 0, "water": 0}) == 0.0


def test_mass_pour_flag_fires_above_threshold_and_cites_aci():
    flag = th.mass_pour_flag(68.2)
    assert flag is not None
    assert "ACI 207" in flag


def test_mass_pour_flag_silent_below_threshold():
    assert th.mass_pour_flag(10.0) is None
    assert th.mass_pour_flag(20.0) is None  # boundary: > threshold, not >=


def test_mass_pour_flag_none_in_none_out():
    assert th.mass_pour_flag(None) is None


# --- C2: equivalent_age (ASTM C1074 Arrhenius maturity) -----------------------------
def test_equivalent_age_at_reference_temperature_is_exact():
    assert th.equivalent_age(28.0, 20.0, ref_c=20.0) == 28.0
    assert th.equivalent_age(1.0, 20.0, ref_c=20.0) == 1.0


def test_equivalent_age_monotonic_in_temperature():
    ages = [th.equivalent_age(28.0, t) for t in (0.0, 10.0, 20.0, 30.0, 40.0)]
    assert ages == sorted(ages)


def test_equivalent_age_35c_vs_20c_ratio_in_expected_band():
    """Arrhenius arithmetic: factor = exp[-Ea/R * (1/T35 - 1/T20)], Ea=41.6 kJ/mol
    (declared ASTM C1074 mid-range default), R=8.314 J/(mol.K).
        T35 = 308.15 K, T20 = 293.15 K
        1/T35 - 1/T20 = -1.660e-4 K^-1
        -Ea/R * (that) = (41600/8.314) * 1.660e-4 = 0.8298
        exp(0.8298) ~= 2.29
    """
    ratio = th.equivalent_age(28.0, 35.0) / th.equivalent_age(28.0, 20.0)
    assert 2.2 <= ratio <= 2.6


def test_curing_days_decreases_with_temperature():
    days = [th.curing_days_at_temperature(REF_MIX, temp_c=t) for t in (5.0, 20.0, 35.0)]
    assert all(d is not None for d in days)
    assert days == sorted(days, reverse=True)


def test_curing_days_at_reference_temperature_equals_direct_alpha_inversion():
    """At temp_c == the model's implicit 20 C reference, the Arrhenius rescale
    factor is 1.0, so curing_days_at_temperature must equal the raw
    alpha(t)-inversion (white-box check against the private inverter)."""
    phases, clinker_factor, cement_mass, w_c, reason = th._hydration_inputs(REF_MIX, "OPC")
    assert reason is None
    alpha_ult = min(1.0, w_c / 0.38)
    direct = th._invert_alpha_to_age(phases, w_c, cement_mass, clinker_factor,
                                      target_alpha=0.7 * alpha_ult)
    via_public_api = th.curing_days_at_temperature(REF_MIX, target_alpha_fraction=0.7, temp_c=20.0)
    assert via_public_api == pytest.approx(direct, rel=1e-9)


def test_curing_days_reaches_target_alpha_fraction():
    """The inverted age really does hit target_alpha_fraction * alpha_ult when
    fed back through hydration_kinetics at the reference temperature."""
    phases, clinker_factor, cement_mass, w_c, _ = th._hydration_inputs(REF_MIX, "OPC")
    alpha_ult = min(1.0, w_c / 0.38)
    te = th.curing_days_at_temperature(REF_MIX, target_alpha_fraction=0.7, temp_c=20.0)
    state = hydration_kinetics(phases, w_c, age_days=te,
                                cement_mass_kg_m3=cement_mass, clinker_factor=clinker_factor)
    assert state.degree_of_hydration == pytest.approx(0.7 * alpha_ult, rel=1e-6)


# --- C3: carbonation upper bound -----------------------------------------------------
def test_carbonation_bound_is_linear_in_ch():
    lo = th.carbonation_co2_bound_kg_m3({"CH_remaining_kg_m3": 10.0})
    hi = th.carbonation_co2_bound_kg_m3({"CH_remaining_kg_m3": 20.0})
    assert hi == pytest.approx(2 * lo, rel=1e-9)
    # Molar ratio CO2/Ca(OH)2 = 44.01/74.09
    assert lo == pytest.approx(10.0 * 44.01 / 74.09, rel=1e-9)


def test_carbonation_bound_matches_real_analyze_mix_output():
    report = analyze_mix(REF_MIX)
    assert report["CH_remaining_kg_m3"] > 0
    bound = th.carbonation_co2_bound_kg_m3(report)
    assert bound == pytest.approx(report["CH_remaining_kg_m3"] * 44.01 / 74.09, rel=1e-9)


def test_carbonation_bound_zero_when_scm_chain_exhausts_ch_pool():
    """Heavy slag dosing: slag's CH draw (LATENT_HYDRAULIC_CH_DRAW=0.05 kg/kg,
    at full reaction extent by age 28 -- see chemistry_advanced.py) against
    900 kg/m3 of slag (900*0.05=45 kg/m3 capacity) exceeds the ~42 kg/m3 CH the
    350 kg/m3 OPC cement produces, so the pool is fully drawn down."""
    heavy_mix = {**REF_MIX, "slag": 900, "coarse_agg": 400, "fine_agg": 400}
    report = analyze_mix(heavy_mix)
    assert report["CH_remaining_kg_m3"] == pytest.approx(0.0, abs=1e-9)
    assert th.carbonation_co2_bound_kg_m3(report) == pytest.approx(0.0, abs=1e-9)


# --- LC3 total-safety: every function degrades, never raises ------------------------
def test_lc3_hydration_is_none_precondition():
    """Precondition the other LC3 tests below rely on: LC3 is bogue_valid=false
    in data/oxide_compositions.json, so analyze_mix reports hydration=None."""
    report = analyze_mix(REF_MIX, cement_type="LC3")
    assert report["hydration"] is None
    assert report["CH_remaining_kg_m3"] == 0.0


def test_adiabatic_rise_degrades_to_none_on_lc3():
    assert th.adiabatic_temperature_rise(REF_MIX, cement_type="LC3") is None


def test_mass_pour_flag_degrades_to_none_on_lc3_chain():
    delta_t = th.adiabatic_temperature_rise(REF_MIX, cement_type="LC3")
    assert th.mass_pour_flag(delta_t) is None


def test_curing_days_degrades_to_none_on_lc3():
    assert th.curing_days_at_temperature(REF_MIX, cement_type="LC3") is None


def test_carbonation_bound_degrades_to_zero_on_lc3():
    report = analyze_mix(REF_MIX, cement_type="LC3")
    assert th.carbonation_co2_bound_kg_m3(report) == 0.0


def test_lc3_chain_never_raises():
    """Belt-and-braces: run the whole WP-C surface against LC3 and an unknown
    cement_type, asserting no exception propagates."""
    for cement_type in ("LC3", "totally-unknown-type"):
        dt = th.adiabatic_temperature_rise(REF_MIX, cement_type=cement_type)
        th.mass_pour_flag(dt)
        th.curing_days_at_temperature(REF_MIX, cement_type=cement_type)
        th.carbonation_co2_bound_kg_m3(analyze_mix(REF_MIX, cement_type=cement_type)
                                        if cement_type == "LC3" else {"CH_remaining_kg_m3": 0.0})


# --- Import boundary (interface contract) --------------------------------------------
def test_no_ui_or_models_imports():
    import inspect
    import src.thermal as thermal_module
    src_text = inspect.getsource(thermal_module)
    assert "from ui" not in src_text
    assert "import ui" not in src_text
    assert "from .models" not in src_text
    assert "from src.models" not in src_text
