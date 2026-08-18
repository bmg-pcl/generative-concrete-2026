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
    simple = cs.calculate_embodied_carbon(OPC_MIX)
    advanced = ca.embodied_carbon_advanced(OPC_MIX)
    # Same boundary => comparable magnitudes (within ~20%), and for high-clinker
    # OPC the simple factor is the more conservative (higher) one.
    assert simple > advanced
    assert abs(simple - advanced) / advanced < 0.20


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
