"""
Tests for R7.5 WP-1 -- hydration/Bogue layer remediation
(see docs/specs/R7.5-chemistry-remediation.md, "WP-1 - Hydration / Bogue layer"):

  * finding 1 -- heat units: `/ 100.0` not `/ 1000.0`; heat at alpha~1 for the
    shipped OPC oxides lands in a physically plausible BAND (350-520 kJ/kg cement).
    This gates units, not a literature match -- see the module docstring in
    src/chemistry_advanced.py for why this layer cannot be calibrated.
  * finding 2 -- one unit system: the whole Bogue -> hydration -> pozzolanic chain
    is kg/m3 (kJ/kg cement for heat); every field name carries its unit.
  * finding 3 -- the portlandite (CH) budget is conserved across SCMs: a running
    pool is threaded through the sequential reactions instead of handing the same
    CH to both, so Sigma CH consumed <= CH produced.
  * finding 4 -- analyze_mix honours cement_type / clinker_source instead of
    hardcoding OPC.
  * finding 5/7 -- Bogue refuses non-Portland (bogue_valid: false) oxide records;
    data/oxide_compositions.json carries the flag on every entry.
  * finding 6/8 -- latent-hydraulic slag gets a distinct reaction path from
    pozzolanic fly ash, with a materially lower CH draw per kg (self-hydration +
    activation, not stoichiometric consumption).

Every gate below is a unit, conservation, or frozen-surface check -- never "matches
the literature value." This layer is dimensionally checked but UNCALIBRATED: there
is no calorimetry or degree-of-hydration data anywhere in this repo to fit the
reaction-rate/stoichiometry coefficients against.
"""
import dataclasses

import pytest

from src import chemistry_advanced as ca

OPC_OXIDES = {"CaO": 65.0, "SiO2": 21.0, "Al2O3": 5.5, "Fe2O3": 3.0, "SO3": 2.5,
              "MgO": 2.0, "bogue_valid": True}

# Pinned Bogue phase output for the shipped OPC oxides -- the textbook equations
# are untouched by WP-1 (see spec "Current state"); this is a migration gate.
EXPECTED_OPC_C3S = pytest.approx(56.646, rel=1e-3)
EXPECTED_OPC_C2S = pytest.approx(17.473, rel=1e-3)
EXPECTED_OPC_C3A = pytest.approx(9.499, rel=1e-3)
EXPECTED_OPC_C4AF = pytest.approx(9.129, rel=1e-3)

# The physically-plausible band for ultimate (alpha -> 1) OPC heat of hydration
# (finding 1's acceptance gate) -- a unit sanity check, not a calibration target.
EXPECTED_HEAT_BAND_KJ_KG_CEMENT = (350.0, 520.0)

BASE_MIX = {"cement": 300, "slag": 0, "ash": 0, "water": 180,
            "coarse_agg": 1000, "fine_agg": 750, "age": 28}


def _opc_phases():
    return ca.bogue_calculation(OPC_OXIDES)


def _full_hydration_state(cement_mass_kg_m3=350.0, clinker_factor=0.95):
    """w_c=1.0 (water-unlimited) and a long age drive degree_of_hydration -> 1."""
    phases = _opc_phases()
    return ca.hydration_kinetics(phases, w_c_ratio=1.0, age_days=3650,
                                  cement_mass_kg_m3=cement_mass_kg_m3,
                                  clinker_factor=clinker_factor)


# --- finding 1: heat units ---------------------------------------------------------
def test_bogue_phases_match_textbook_values():
    p = _opc_phases()
    assert p.C3S == EXPECTED_OPC_C3S
    assert p.C2S == EXPECTED_OPC_C2S
    assert p.C3A == EXPECTED_OPC_C3A
    assert p.C4AF == EXPECTED_OPC_C4AF


def test_heat_at_full_hydration_is_physically_plausible_band():
    state = _full_hydration_state()
    assert state.degree_of_hydration == pytest.approx(1.0, abs=1e-6)
    lo, hi = EXPECTED_HEAT_BAND_KJ_KG_CEMENT
    assert lo <= state.heat_kJ_per_kg_cement <= hi


def test_heat_is_not_the_old_ten_x_low_value():
    """Regression guard for the factor-of-10 unit bug: the pre-fix `/ 1000.0`
    would have landed near 49 kJ/kg cement at alpha~1 -- an order of magnitude
    below the plausible band."""
    state = _full_hydration_state()
    assert state.heat_kJ_per_kg_cement > 100.0


# --- finding 2: one unit system (kg/m3 throughout) ----------------------------------
def test_hydration_state_fields_carry_units():
    names = {f.name for f in dataclasses.fields(ca.HydrationState)}
    assert names == {"age_days", "degree_of_hydration", "CSH_kg_m3", "CH_kg_m3",
                      "heat_kJ_per_kg_cement"}


def test_ch_and_csh_scale_with_cement_mass_not_bare_wt_pct():
    """CH/CSH are kg/m3 of concrete: doubling cement dosage (fixed phase
    composition and clinker_factor) doubles the CH/CSH mass -- a bare wt%
    quantity would not do this, which is exactly the old defect."""
    phases = _opc_phases()
    small = ca.hydration_kinetics(phases, w_c_ratio=0.5, age_days=28,
                                   cement_mass_kg_m3=300.0, clinker_factor=0.95)
    large = ca.hydration_kinetics(phases, w_c_ratio=0.5, age_days=28,
                                   cement_mass_kg_m3=600.0, clinker_factor=0.95)
    assert large.CH_kg_m3 == pytest.approx(2 * small.CH_kg_m3, rel=1e-9)
    assert large.CSH_kg_m3 == pytest.approx(2 * small.CSH_kg_m3, rel=1e-9)


def test_pozzolanic_reaction_units_are_kg_m3():
    consumed, csh = ca.pozzolanic_reaction(CH_available_kg_m3=50.0,
                                            pozzolan_mass_kg_m3=100.0,
                                            pozzolan_type="FLY_ASH_F", age_days=28)
    assert 0.0 <= consumed <= 50.0
    assert csh == pytest.approx(consumed * 1.5)


def test_analyze_mix_return_keys_carry_units():
    mix = {**BASE_MIX, "slag": 50, "ash": 20}
    report = ca.analyze_mix(mix)
    for key in ("total_CSH_kg_m3", "carbon_kg_m3", "pozzolanic_CSH_kg_m3",
                "CH_remaining_kg_m3"):
        assert key in report


# --- finding 3: conserve the portlandite budget -------------------------------------
def test_ch_budget_conserved_heavy_dosing_exhausts_pool():
    """Sigma CH consumed <= CH produced, with equality only when heavy dosing of
    BOTH SCMs exhausts the pool -- the defect-2 regression gate."""
    mix = {**BASE_MIX, "slag": 400, "ash": 400, "age": 90}
    report = ca.analyze_mix(mix)
    ch_produced = report["hydration"].CH_kg_m3
    ch_consumed = ch_produced - report["CH_remaining_kg_m3"]
    assert ch_consumed <= ch_produced + 1e-9
    assert report["CH_remaining_kg_m3"] >= -1e-9


def test_ch_budget_not_double_spent_light_dosing():
    """With light dosing (pool not exhausted), adding ash on top of slag can only
    consume MORE cumulative CH than slag alone, never double-count the same CH
    against both SCMs independently (the pre-fix defect: each SCM drew against
    the FULL pool)."""
    mix_slag_only = {**BASE_MIX, "slag": 20, "ash": 0}
    mix_both = {**BASE_MIX, "slag": 20, "ash": 20}
    r_slag = ca.analyze_mix(mix_slag_only)
    r_both = ca.analyze_mix(mix_both)
    ch_produced = r_slag["hydration"].CH_kg_m3
    slag_only_consumed = ch_produced - r_slag["CH_remaining_kg_m3"]
    both_consumed = ch_produced - r_both["CH_remaining_kg_m3"]
    assert both_consumed >= slag_only_consumed
    assert both_consumed <= ch_produced + 1e-9


# --- finding 4: honour cement_type / clinker_source in analyze_mix ------------------
def test_analyze_mix_honours_cement_type_for_carbon():
    opc = ca.analyze_mix(BASE_MIX, cement_type="OPC")
    lc3 = ca.analyze_mix(BASE_MIX, cement_type="LC3")
    assert lc3["carbon_kg_m3"] < opc["carbon_kg_m3"]


def test_analyze_mix_honours_clinker_source():
    base = ca.analyze_mix(BASE_MIX)
    clean = ca.analyze_mix(
        BASE_MIX, clinker_source={"kiln_fuel": "electric", "electricity": "hydro"})
    assert clean["carbon_kg_m3"] < base["carbon_kg_m3"]


# --- finding 5/7: refuse Bogue on blended cements -----------------------------------
def test_bogue_calculation_raises_on_bogue_invalid_record():
    with pytest.raises(ValueError):
        ca.bogue_calculation({"CaO": 55.0, "SiO2": 25.0, "Al2O3": 10.0, "Fe2O3": 3.0,
                              "bogue_valid": False})


def test_bogue_calculation_permits_record_with_no_flag():
    """A bare oxide dict with no bogue_valid key (not sourced from the registry)
    is treated as valid: the flag is an explicit 'no', not a required 'yes'."""
    phases = ca.bogue_calculation({"CaO": 65.0, "SiO2": 21.0, "Al2O3": 5.5,
                                    "Fe2O3": 3.0, "SO3": 2.5})
    assert phases.C3S > 0


def test_analyze_mix_lc3_does_not_return_opc_phase_values():
    opc_phases = ca.analyze_mix(BASE_MIX, cement_type="OPC")["clinker_phases"]
    lc3_report = ca.analyze_mix(BASE_MIX, cement_type="LC3")
    assert lc3_report["clinker_phases"] is None
    assert lc3_report["hydration"] is None
    assert lc3_report["phases_unavailable_reason"] is not None
    assert opc_phases is not None  # sanity: OPC path still works


def test_analyze_mix_unknown_cement_type_reports_unavailable_not_crash():
    report = ca.analyze_mix(BASE_MIX, cement_type="does_not_exist")
    assert report["clinker_phases"] is None
    assert "does_not_exist" in report["phases_unavailable_reason"]


def test_oxide_json_bogue_valid_flags():
    comps = ca.load_oxide_compositions()
    assert comps["OPC"]["bogue_valid"] is True
    for blended in ("SLAG", "FLY_ASH_F", "FLY_ASH_C", "LC3"):
        assert comps[blended]["bogue_valid"] is False


# --- finding 6/8: latent-hydraulic (slag) vs pozzolanic (fly ash) ------------------
def test_slag_draws_less_ch_per_kg_than_fly_ash():
    """Latent-hydraulic slag activates rather than stoichiometrically consumes CH,
    so at equal mass/age its CH draw must be far below fly ash's pozzolanic draw."""
    ch_pool = 1000.0  # effectively unlimited, so both hit their own ceiling
    slag_consumed, _ = ca.latent_hydraulic_reaction(ch_pool, 200.0, "SLAG", 28)
    ash_consumed, _ = ca.pozzolanic_reaction(ch_pool, 200.0, "FLY_ASH_F", 28)
    assert slag_consumed < ash_consumed


def test_analyze_mix_routes_slag_through_latent_hydraulic_path():
    """data/materials.json marks slag's reactivity_class as latent-hydraulic and
    ash's as pozzolanic; analyze_mix must read that distinction, not treat both
    SCMs identically."""
    mix = {**BASE_MIX, "slag": 100, "ash": 0}
    report = ca.analyze_mix(mix)
    ch_produced = report["hydration"].CH_kg_m3
    ch_consumed = ch_produced - report["CH_remaining_kg_m3"]
    expected_consumed, expected_csh = ca.latent_hydraulic_reaction(
        ch_produced, 100.0, "SLAG", 28)
    assert ch_consumed == pytest.approx(expected_consumed)
    assert report["pozzolanic_CSH_kg_m3"] == pytest.approx(expected_csh)


# --- frozen surface: WP-1 must not perturb the carbon tables/functions -------------
def test_frozen_carbon_constants_unchanged():
    assert ca.CALCINATION_EF == 0.53
    assert ca.FUEL_EF == {"coal": 0.40, "petcoke": 0.42, "natural_gas": 0.28,
                          "alt_waste": 0.15, "biomass": 0.05, "electric": 0.0}
    assert ca.GRID_EF == {"grid_EU": 0.25, "grid_US": 0.35, "grid_CN": 0.55,
                          "hydro": 0.004, "nuclear": 0.012, "ppa_renewable": 0.02}
    assert ca.KILN_KWH_PER_KG == {"electric": 0.95}


# --- traps: Bogue clamp ordering (C2S from unclamped C3S) must survive -------------
def test_bogue_c2s_uses_unclamped_c3s():
    """A record whose raw C3S exceeds the 80% clamp must still compute C2S from
    the RAW value, not the clamped one -- the clamp is presentational only, and
    reordering it would silently change C2S."""
    oxides = {"CaO": 79.48, "SiO2": 25.0, "Al2O3": 5.0, "Fe2O3": 3.0, "SO3": 2.0,
              "bogue_valid": True}
    raw_c3s = (4.071 * oxides["CaO"] - 7.600 * oxides["SiO2"] - 6.718 * oxides["Al2O3"]
               - 1.430 * oxides["Fe2O3"] - 2.852 * oxides["SO3"])
    assert raw_c3s > 80  # otherwise this test isn't exercising the clamp trap
    expected_c2s = max(0, min(50, 2.867 * oxides["SiO2"] - 0.7544 * raw_c3s))

    phases = ca.bogue_calculation(oxides)
    assert phases.C3S == 80  # clamped
    assert phases.C2S == pytest.approx(expected_c2s)
