"""R6.3 — clinker source differentiation tests.

The worked-example table from docs/specs/R6-materials-platform.md §4.2, plus the two
traps the spec calls out: the legacy default must be byte-identical (no descriptor →
0.53 + 0.35), and capture's energy penalty must SURVIVE capture (it is charged to the
chosen electricity, not captured along with the stack).
"""
import pytest

from src.chemistry_advanced import (
    CALCINATION_EF,
    carbon_from_clinker,
    clinker_scope_split,
    embodied_carbon_advanced,
)
from src.ui_logic import carbon_for_mode, carbon_breakdown, mix_dict, mix_ticket

MIX = mix_dict([350, 100, 0, 175, 5, 1000, 750, 28])


def _per_kg(source):
    return clinker_scope_split(1.0, source)


# --- the spec §4.2 worked-example table (±10%) -----------------------------------
def test_standard_coal_kiln():
    assert _per_kg({"kiln_fuel": "coal", "electricity": "grid_EU"})["total"] == \
        pytest.approx(0.93, rel=0.10)


def test_standard_natural_gas_kiln():
    assert _per_kg({"kiln_fuel": "natural_gas", "electricity": "grid_EU"})["total"] == \
        pytest.approx(0.81, rel=0.10)


def test_capture_on_grid_vs_hydro():
    grid = _per_kg({"kiln_fuel": "natural_gas", "electricity": "grid_EU",
                    "capture": {"rate": 0.90}})
    hydro = _per_kg({"kiln_fuel": "natural_gas", "electricity": "hydro",
                     "capture": {"rate": 0.90}})
    assert grid["total"] == pytest.approx(0.11, rel=0.10)
    assert hydro["total"] == pytest.approx(0.08, rel=0.10)
    # Same plant, same capture — the difference is WHERE the capture energy comes from.
    assert hydro["total"] < grid["total"]
    assert grid["scope1"] == pytest.approx(hydro["scope1"])


def test_electrified_kiln_on_hydro_leaves_calcination():
    split = _per_kg({"kiln_fuel": "electric", "electricity": "hydro"})
    assert split["total"] == pytest.approx(0.53, rel=0.10)
    assert split["scope1"] == pytest.approx(CALCINATION_EF)  # only process CO2 remains


def test_electrified_hydro_plus_capture_is_the_floor():
    split = _per_kg({"kiln_fuel": "electric", "electricity": "hydro",
                     "capture": {"rate": 0.90}})
    assert split["total"] == pytest.approx(0.06, rel=0.15)


# --- the traps -------------------------------------------------------------------
def test_capture_energy_penalty_survives_capture():
    """The spec's inversion trap: capture applies to (process + combustion), NOT to
    its own Scope 2 energy — even at maximal capture, Scope 2 stays positive on a
    fossil grid and grows with the capture rate."""
    lo = _per_kg({"kiln_fuel": "natural_gas", "electricity": "grid_EU",
                  "capture": {"rate": 0.50}})
    hi = _per_kg({"kiln_fuel": "natural_gas", "electricity": "grid_EU",
                  "capture": {"rate": 0.95}})
    assert hi["scope2"] > lo["scope2"] > 0.0
    assert hi["scope1"] < lo["scope1"]


def test_legacy_default_unchanged():
    # R8.0/WP-B B1: the legacy (no-descriptor) path now also carries the process-
    # electricity term (previously dropped scope -- see chemistry_advanced module
    # docstring), on the documented legacy grid_EU default:
    #   clinker term = 300 * 0.95 * (0.53 + 0.35)        = 250.8
    #   process term = 300 * 0.11 * GRID_EF["grid_EU"]   = 300 * 0.11 * 0.25 = 8.25
    #   total                                            = 259.05  (~0.8635/kg cement)
    # OPC has no `constituents` entry, so B2 contributes exactly 0 here -- this
    # gate is the "at least one gate runs the legacy path" requirement (grid_EU
    # fallback coverage) from the spec's WP-B "Traps" note.
    legacy_clinker_term = 300.0 * 0.95 * (0.53 + 0.35)
    legacy_process_term = 300.0 * 0.11 * 0.25
    assert carbon_from_clinker(300.0) == pytest.approx(
        legacy_clinker_term + legacy_process_term)
    assert embodied_carbon_advanced(MIX) == pytest.approx(
        embodied_carbon_advanced(MIX, clinker_source=None))


# --- integration: one carbon path, breakdown reconciles, ticket discloses ---------
SOURCE = {"kiln_fuel": "natural_gas", "electricity": "hydro", "capture": {"rate": 0.9}}


def test_clinker_source_lowers_advanced_carbon_and_reconciles():
    base = carbon_for_mode(MIX, advanced=True)
    low = carbon_for_mode(MIX, advanced=True, clinker_source=SOURCE)
    assert low < base
    bd = carbon_breakdown(MIX, advanced=True, clinker_source=SOURCE, transport_km=50.0)
    assert sum(bd.values()) == pytest.approx(
        carbon_for_mode(MIX, advanced=True, clinker_source=SOURCE, transport_km=50.0))


def test_ticket_reports_scope_split_and_provenance():
    from src.materials import carbon_provenance, effective_carbon_factors
    epds = {"cement": {"value": 0.55, "reference": "EPD-BREVIK-2025"}}
    factors = effective_carbon_factors(epds)
    metrics = {"strength": 45.0, "interval_lo": 40.0, "interval_hi": 50.0,
               "tensile": 3.8, "curing": 9.0, "novelty": 0.8, "in_support": True}
    config = {"advanced": True, "transport_km": 0.0, "cement_type": "OPC",
              "factors": factors, "clinker_source": SOURCE, "costs": None,
              "robust": True, "carbon_provenance": carbon_provenance(factors, epds),
              "timestamp": "2026-07-02T00:00:00+00:00"}
    csv = mix_ticket(MIX, metrics, config)
    lines = csv.splitlines()
    assert any(line.startswith('provenance,cement,"epd:EPD-BREVIK-2025"') for line in lines)
    assert any(line.startswith("provenance,slag,") for line in lines)
    assert any(line.startswith("clinker_scope,scope1_kgCO2,") for line in lines)
    assert any(line.startswith("clinker_scope,scope2_kgCO2,") for line in lines)
    assert any(line == "clinker_scope,capture_rate,0.9" for line in lines)
    # The TOTAL row still reconciles to the single carbon path under the descriptor.
    total = float(next(line for line in lines
                       if line.startswith("carbon_kgCO2,TOTAL,")).split(",")[2])
    assert total == pytest.approx(
        carbon_for_mode(MIX, advanced=True, factors=factors, clinker_source=SOURCE),
        abs=0.05)
