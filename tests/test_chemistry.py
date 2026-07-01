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
