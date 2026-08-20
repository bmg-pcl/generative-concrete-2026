"""
Tests for the Phase 2 chemistry-correctness fixes (see docs/FIX_PLAN.md):
  * the advanced carbon tier is a SUPERSET of its clinker term and lives on the
    SAME system boundary as the simple tier (fidelity, not scope);
  * clinker factors are read from data/oxide_compositions.json (OPC vs LC3);
  * the simple tier runs slightly higher than advanced for high-clinker OPC
    (this is the corrected direction; the report used to claim the opposite).
"""
import pytest

from src import chemistry_simple as cs
from src import chemistry_advanced as ca

OPC_MIX = {
    "cement": 400, "slag": 0, "ash": 0, "water": 180,
    "superplasticizer": 3, "coarse_agg": 1000, "fine_agg": 750,
}


def test_clinker_factor_from_json():
    assert ca.clinker_factor_for("OPC") == pytest.approx(0.95)
    assert ca.clinker_factor_for("LC3") == pytest.approx(0.50)
    # Unknown type falls back to the OPC-like default.
    assert ca.clinker_factor_for("does_not_exist") == pytest.approx(0.95)


def test_advanced_is_superset_of_clinker_term():
    total = ca.embodied_carbon_advanced(OPC_MIX)
    clinker_only = ca.carbon_from_clinker(OPC_MIX["cement"])
    # The full-mix figure includes the clinker term plus non-cement constituents.
    assert total > clinker_only


def test_tiers_same_boundary_simple_slightly_higher():
    """R8.0/WP-B B1 re-derivation: process electricity (0.11 kWh/kg cement x
    GRID_EF["grid_EU"]=0.25 = 0.0275/kg) closes most of what used to be dropped
    scope in the advanced cement term, so the simple-vs-advanced gap for
    high-clinker OPC narrows a lot (was ~8.7%, now ~5.4% -- both derived below):

        clinker term/kg  = 0.95 * (0.53 + 0.35)              = 0.8360
        process term/kg  = 0.11 * 0.25                        = 0.0275
        advanced cement term/kg (legacy grid)                 = 0.8635  (~0.864)

    OPC_MIX advanced total = 361.704, simple total = 381.104 (simple's cement
    factor is 0.912/kg, chemistry_simple.CARBON_FACTORS["cement"] -- unowned,
    unchanged by WP-B) => relative gap = |381.104-361.704|/361.704 ~= 0.0536.
    Tier ordering is preserved (simple still the more conservative, higher
    figure) but the tolerance is re-derived down from the old 20% ceiling to
    bound the now-narrower gap without loosening it blindly.
    """
    simple = cs.calculate_embodied_carbon(OPC_MIX)
    advanced = ca.embodied_carbon_advanced(OPC_MIX)
    assert simple > advanced
    gap = abs(simple - advanced) / advanced
    assert gap == pytest.approx(0.0536, abs=0.002)
    assert gap < 0.10


def test_lc3_lowers_carbon():
    opc = ca.embodied_carbon_advanced(OPC_MIX, cement_type="OPC")
    lc3 = ca.embodied_carbon_advanced(OPC_MIX, cement_type="LC3")
    assert lc3 < opc


def test_transport_adds_carbon():
    base = ca.embodied_carbon_advanced(OPC_MIX)
    shipped = ca.embodied_carbon_advanced(OPC_MIX, transport_km=500)
    assert shipped > base


def test_transport_ignores_age():
    """Age is a curing time, not a mass, and must not inflate transport carbon."""
    young = {**OPC_MIX, "age": 1}
    old = {**OPC_MIX, "age": 365}
    assert (ca.embodied_carbon_advanced(young, transport_km=500)
            == ca.embodied_carbon_advanced(old, transport_km=500))
    assert (cs.calculate_embodied_carbon(young, transport_km=500)
            == cs.calculate_embodied_carbon(old, transport_km=500))


# --- R7.5 WP-5: transport mass consistency ---------------------------------------
# The transport heuristic ((total_mass/1000) * km * 0.1) used to be triplicated
# across chemistry_simple.py, chemistry_advanced.py, and ui_logic.py, and summed
# only the seven core factor keys -- so exotic admixture mass shipped for free (a
# small, systematic, always-low bias). Now there is ONE definition,
# `chemistry_simple.transport_carbon`, shared by both tiers and by
# `ui_logic.carbon_breakdown`.

def test_transport_carbon_is_the_single_shared_definition():
    """Both tiers delegate to the same `transport_carbon` function (not just to
    equivalent-looking arithmetic) -- chemistry_advanced imports it directly."""
    assert ca.transport_carbon is cs.transport_carbon


def test_exotic_mass_raises_transport_term_both_tiers():
    """A mix with exotics dosed shows a HIGHER transport term than the same mix
    without -- and the delta matches the exact expected value: 100 kg/m3 dosed x
    500 km x 0.1/1000 = 5.0 kg CO2 (R7.5 WP-5 acceptance gate)."""
    exotic = {"silica_fume": 100.0}
    expected_delta = (100.0 / 1000.0) * 500.0 * 0.1
    assert expected_delta == pytest.approx(5.0)

    simple_base = cs.calculate_embodied_carbon(OPC_MIX, transport_km=500)
    simple_dosed = cs.calculate_embodied_carbon(OPC_MIX, transport_km=500, exotic=exotic)
    assert simple_dosed - simple_base == pytest.approx(expected_delta)

    adv_base = ca.embodied_carbon_advanced(OPC_MIX, transport_km=500)
    adv_dosed = ca.embodied_carbon_advanced(OPC_MIX, transport_km=500, exotic=exotic)
    assert adv_dosed - adv_base == pytest.approx(expected_delta)


def test_exotic_none_default_is_bit_identical_to_pre_wp5():
    """Default exotic=None must not move a single number (both tiers)."""
    assert (cs.calculate_embodied_carbon(OPC_MIX, transport_km=500)
            == cs.calculate_embodied_carbon(OPC_MIX, transport_km=500, exotic=None))
    assert (ca.embodied_carbon_advanced(OPC_MIX, transport_km=500)
            == ca.embodied_carbon_advanced(OPC_MIX, transport_km=500, exotic=None))
    # An empty exotic dict must also be a no-op (falsy -- same as None).
    assert (cs.calculate_embodied_carbon(OPC_MIX, transport_km=500)
            == cs.calculate_embodied_carbon(OPC_MIX, transport_km=500, exotic={}))


def test_exotic_mass_still_ignores_age():
    """'age' must stay excluded from transport mass even when an exotic dict is
    also passed, in both tiers."""
    exotic = {"silica_fume": 50.0}
    young = {**OPC_MIX, "age": 1}
    old = {**OPC_MIX, "age": 365}
    assert (cs.calculate_embodied_carbon(young, transport_km=500, exotic=exotic)
            == cs.calculate_embodied_carbon(old, transport_km=500, exotic=exotic))
    assert (ca.embodied_carbon_advanced(young, transport_km=500, exotic=exotic)
            == ca.embodied_carbon_advanced(old, transport_km=500, exotic=exotic))


# --- R7.5 WP-3: the curing heuristic no longer double-penalises SCMs -------------
# It used water/CEMENT *and* an unbounded (ash+slag)/cement term, charging a
# cement->SCM substitution twice for the same swap. Now: water/BINDER for dilution,
# plus a bounded SCM fraction for the slower reaction rate. See spec R7.5 WP-3.

SCM_TERM_MAX = 5.0   # scm_fraction is bounded in [0, 1] and scaled by 5 days


def test_scm_substitution_is_charged_once_not_twice():
    blended = cs.estimate_curing_time(
        {"cement": 200, "slag": 200, "ash": 0, "water": 180})
    plain = cs.estimate_curing_time(
        {"cement": 400, "slag": 0, "ash": 0, "water": 180})
    # Same binder (400) and same water => same w/b; the ONLY legitimate difference is
    # the bounded SCM rate term. The old formula gave 17.0 vs 7.5 -- a 9.5-day gap
    # off an apparent w/c of 0.9, when the true w/b is 0.45 for both.
    assert blended - plain <= SCM_TERM_MAX + 1e-9
    assert blended > plain, "SCMs should still cure slower -- just not doubly so"


def test_curing_is_monotonic_in_water_binder_ratio():
    def days(water):
        return cs.estimate_curing_time(
            {"cement": 400, "slag": 0, "ash": 0, "water": water})
    assert days(140) < days(180) < days(220)


def test_curing_uses_binder_not_cement_for_dilution():
    """Two mixes with identical binder and water have identical w/b, so they may
    differ only by the SCM term -- not by a cement-denominated ratio."""
    a = cs.estimate_curing_time({"cement": 400, "slag": 0, "ash": 0, "water": 180})
    b = cs.estimate_curing_time({"cement": 390, "slag": 10, "ash": 0, "water": 180})
    assert abs((b - a) - (10 / 400) * SCM_TERM_MAX) < 1e-9


def test_curing_floor_and_zero_binder_are_safe():
    # The >= 1 day floor is a guarantee across the whole reachable input range. Note
    # it is DEFENSIVE, not load-bearing: with water >= 0 the raw formula bottoms out
    # at 7 + (0 - 0.4) * 10 = 3.0 days, so max(1.0, ...) never actually binds. Kept
    # so a future coefficient change cannot silently emit a nonsensical value.
    for mix in ({"cement": 900, "slag": 0, "ash": 0, "water": 100},
                {"cement": 550, "slag": 0, "ash": 0, "water": 0},
                {"cement": 100, "slag": 200, "ash": 200, "water": 250}):
        assert cs.estimate_curing_time(mix) >= 1.0
    # No binder at all must not raise ZeroDivisionError.
    assert cs.estimate_curing_time(
        {"cement": 0, "slag": 0, "ash": 0, "water": 0}) >= 1.0


# --- R8.0 / WP-B: Tier-2 cement-term scope completion -----------------------------
# B1 (process electricity) + B2 (LC3 constituent carbon) close scope that was
# previously dropped from the advanced cement term; B3 tightens Bogue's bare-dict
# hygiene. See docs/specs/R8.0-carbon-chemistry-deepening.md, WP-B.

def test_process_electricity_raises_legacy_opc_cement_term():
    """B1: legacy (no clinker_source) OPC cement term now includes process
    electricity on the documented GRID_EF['grid_EU'] default. Per kg cement:
        clinker term/kg = 0.95 * (0.53 + 0.35) = 0.8360
        process term/kg = 0.11 * 0.25          = 0.0275
        total/kg                               = 0.8635  (spec's ~0.864/kg)
    The term is linear in cement_mass, so the per-kg figure is mass-independent;
    checked at both 400 kg/m3 (absolute) and 300 kg/m3 (per-kg, matching the
    spec's worked example).
    """
    clinker_term = 400.0 * 0.95 * (0.53 + 0.35)
    process_term = 400.0 * 0.11 * ca.GRID_EF["grid_EU"]
    assert ca.carbon_from_clinker(400.0) == pytest.approx(clinker_term + process_term)
    per_kg = ca.carbon_from_clinker(300.0) / 300.0
    assert per_kg == pytest.approx(0.8635, abs=1e-3)


def test_process_electricity_zero_for_electrified_hydro_kiln_stays_tiny():
    """A fully electrified, hydro-powered kiln (near-zero grid EF) still pays the
    B1 process-electricity term on whatever grid the descriptor names -- it is not
    folded into (and does not disappear with) clinker_scope_split's own terms."""
    source = {"kiln_fuel": "electric", "electricity": "hydro"}
    clinker_only = ca.clinker_scope_split(300.0 * 0.95, source)["total"]
    total = ca.carbon_from_clinker(300.0, clinker_source=source)
    process_term = 300.0 * ca.CEMENT_PROCESS_KWH_PER_KG * ca.GRID_EF["hydro"]
    assert total == pytest.approx(clinker_only + process_term)
    assert process_term > 0.0  # small, but not zero -- hydro isn't free electricity


def test_descriptor_grid_selects_process_electricity_grid():
    """With a clinker_source descriptor, B1's process term uses THAT descriptor's
    electricity grid (not the legacy grid_EU default): hydro vs grid_CN differ by
    exactly 0.11 * (GRID_EF['grid_CN'] - GRID_EF['hydro']) * cement_mass, since the
    two descriptors share every other term (same kiln_fuel, no capture)."""
    cement_mass = 300.0
    hydro = ca.carbon_from_clinker(
        cement_mass, clinker_source={"kiln_fuel": "natural_gas", "electricity": "hydro"})
    grid_cn = ca.carbon_from_clinker(
        cement_mass, clinker_source={"kiln_fuel": "natural_gas", "electricity": "grid_CN"})
    expected_diff = 0.11 * (ca.GRID_EF["grid_CN"] - ca.GRID_EF["hydro"]) * cement_mass
    assert grid_cn - hydro == pytest.approx(expected_diff)


def test_lc3_constituent_carbon_lands_in_published_epd_band():
    """B2 external-validation gate (the spec's ONE such gate in this wave): the
    LC3 advanced cement term, on the legacy grid, lands in the published LC3 EPD
    band [0.50, 0.65] kg CO2/kg cement -- derived, not fit:
        clinker term/kg      = 0.50 * 0.88                          = 0.4400
        calcined_clay/kg     = 0.30 * 0.25 (materials.json A1-A3)   = 0.0750
        limestone_filler/kg  = 0.15 * 0.01 (materials.json A1-A3)   = 0.0015
        process term/kg      = 0.11 * 0.25                          = 0.0275
        total/kg                                                   = 0.5440
    (~0.545 as sketched in the spec; the ~5% gypsum fraction is documented-excluded,
    see the CEMENT_PROCESS_KWH_PER_KG-adjacent comment in chemistry_advanced.py).
    """
    cement_mass = 400.0
    term = ca.carbon_from_clinker(cement_mass, cement_type="LC3")
    per_kg = term / cement_mass
    assert 0.50 <= per_kg <= 0.65
    assert per_kg == pytest.approx(0.544, abs=0.002)


def test_lc3_constituents_recipe_present_and_opc_has_none():
    comps = ca.load_oxide_compositions()
    assert comps["LC3"]["constituents"] == {
        "calcined_clay": 0.30, "limestone_filler": 0.15}
    assert "constituents" not in comps["OPC"]


def test_opc_unaffected_by_b2_no_constituents_key():
    """OPC has no `constituents` entry, so B2 contributes exactly 0 -- the OPC
    cement term is B1-only relative to the pre-WP-B legacy formula."""
    cement_mass = 250.0
    legacy_plus_b1 = (cement_mass * 0.95 * (0.53 + 0.35)
                      + cement_mass * ca.CEMENT_PROCESS_KWH_PER_KG * ca.GRID_EF["grid_EU"])
    assert ca.carbon_from_clinker(cement_mass, cement_type="OPC") == pytest.approx(
        legacy_plus_b1)


# --- B3: Bogue bare-dict hygiene ---------------------------------------------------
def test_bogue_bare_dict_missing_cao_and_sio2_raises_naming_both():
    with pytest.raises(ValueError) as exc:
        ca.bogue_calculation({})
    msg = str(exc.value)
    assert "CaO" in msg
    assert "SiO2" in msg


def test_bogue_bare_dict_missing_sio2_only_raises_naming_it():
    with pytest.raises(ValueError) as exc:
        ca.bogue_calculation({"CaO": 65})
    msg = str(exc.value)
    assert "SiO2" in msg
    # CaO was supplied, so it must not appear in the missing-oxide LIST -- the
    # message's general explanatory text mentions "CaO and SiO2" together, so
    # check the specific "missing required major oxide(s) ..." clause only.
    missing_clause = msg.split("major oxide(s)")[1].split("--")[0]
    assert "CaO" not in missing_clause
    assert "SiO2" in missing_clause


def test_bogue_registry_record_path_bit_identical_after_b3():
    """Every registry record (data/oxide_compositions.json) carries CaO and SiO2,
    so the registry-record path through analyze_mix is byte-identical after B3 --
    only the bare-dict path (missing majors) is tightened."""
    comps = ca.load_oxide_compositions()
    phases = ca.bogue_calculation(comps["OPC"])
    assert phases.C3S == pytest.approx(56.646, rel=1e-3)
    assert phases.C2S == pytest.approx(17.473, rel=1e-3)
    assert phases.C3A == pytest.approx(9.499, rel=1e-3)
    assert phases.C4AF == pytest.approx(9.129, rel=1e-3)
