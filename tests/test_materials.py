"""R6 — material registry tests.

R6.1 gate: the legacy dicts, now views over data/materials.json, must be
BIT-IDENTICAL to the pre-registry literals (pinned below). R6.2: EPD plug-in and
provenance resolution. R6.4: a registry JSON edit (no code change) surfaces a new
material in every view — the pluggability contract.
"""
import json

import pytest

from src.materials import (
    load_materials,
    set_materials_path,
    validate_material,
    validate_epd_json,
    effective_carbon_factors,
    carbon_provenance,
    carbon_factors_view,
    unit_costs_view,
    densities_view,
    exotics_view,
)

# --- R6.1: pinned pre-registry values (the migration must not move a digit) -----
EXPECTED_CARBON = {"cement": 0.912, "slag": 0.052, "ash": 0.004, "water": 0.0003,
                   "superplasticizer": 1.5, "coarse_agg": 0.008, "fine_agg": 0.005}
EXPECTED_COSTS = {"cement": 0.15, "slag": 0.08, "ash": 0.05, "water": 0.002,
                  "superplasticizer": 2.50, "coarse_agg": 0.03, "fine_agg": 0.04}
EXPECTED_DENSITIES = {"cement": 3150.0, "slag": 2900.0, "ash": 2300.0, "water": 1000.0,
                      "superplasticizer": 1100.0, "coarse_agg": 2700.0, "fine_agg": 2650.0}
EXPECTED_EXOTICS = {
    "silica_fume":         {"default": 0, "max": 50,  "carbon_factor": 0.02, "cost": 0.80,   "category": "Pozzolan", "strength_factor": 0.08},
    "metakaolin":          {"default": 0, "max": 80,  "carbon_factor": 0.30, "cost": 0.45,   "category": "Pozzolan", "strength_factor": 0.06},
    "rice_husk_ash":       {"default": 0, "max": 60,  "carbon_factor": 0.01, "cost": 0.15,   "category": "Pozzolan", "strength_factor": 0.04},
    "limestone_filler":    {"default": 0, "max": 100, "carbon_factor": 0.01, "cost": 0.05,   "category": "Filler",   "strength_factor": -0.01},
    "calcined_clay":       {"default": 0, "max": 150, "carbon_factor": 0.25, "cost": 0.12,   "category": "Filler",   "strength_factor": 0.03},
    "steel_fiber":         {"default": 0, "max": 80,  "carbon_factor": 1.80, "cost": 1.50,   "category": "Fiber",    "strength_factor": 0.05},
    "polypropylene_fiber": {"default": 0, "max": 10,  "carbon_factor": 3.50, "cost": 4.00,   "category": "Fiber",    "strength_factor": -0.02},
    "basalt_fiber":        {"default": 0, "max": 20,  "carbon_factor": 0.60, "cost": 2.50,   "category": "Fiber",    "strength_factor": 0.02},
    "nano_silica":         {"default": 0, "max": 5,   "carbon_factor": 5.00, "cost": 25.00,  "category": "Nano",     "strength_factor": 1.50},
    "graphene_oxide":      {"default": 0, "max": 1,   "carbon_factor": 50.00, "cost": 500.00, "category": "Nano",     "strength_factor": 8.00},
    "calcium_chloride":    {"default": 0, "max": 10,  "carbon_factor": 0.80, "cost": 0.30,   "category": "Chemical", "strength_factor": 0.10},
    "shrink_reducer":      {"default": 0, "max": 8,   "carbon_factor": 2.00, "cost": 6.00,   "category": "Chemical", "strength_factor": 0.00},
}


def test_views_are_bit_identical_to_pre_registry_literals():
    assert carbon_factors_view() == EXPECTED_CARBON
    assert unit_costs_view() == EXPECTED_COSTS
    assert densities_view() == EXPECTED_DENSITIES
    assert exotics_view() == EXPECTED_EXOTICS
    # Iteration ORDER matters too (the UI editors iterate these dicts).
    assert list(carbon_factors_view()) == list(EXPECTED_CARBON)
    assert list(exotics_view()) == list(EXPECTED_EXOTICS)


def test_legacy_module_dicts_are_the_views():
    from src.chemistry_simple import CARBON_FACTORS, UNIT_COSTS
    from src.physical import DENSITIES
    from src.exotics import EXOTIC_ADMIXTURES
    assert CARBON_FACTORS == EXPECTED_CARBON
    assert UNIT_COSTS == EXPECTED_COSTS
    assert DENSITIES == EXPECTED_DENSITIES
    assert EXOTIC_ADMIXTURES == EXPECTED_EXOTICS


def test_every_material_validates_and_carries_carbon():
    for key, rec in load_materials().items():
        assert validate_material(key, rec) is None
        assert rec["carbon"], f"{key} has no carbon record"
        first = rec["carbon"][0]
        assert {"value", "boundary", "source", "reference"} <= set(first)


def test_validate_material_rejects_bad_records():
    assert validate_material("x", {"name": "X"}) is not None
    assert "carbon record" in validate_material(
        "x", {"name": "X", "category": "scm", "carbon": [],
              "strength_treatment": "inert"})
    assert "strength_treatment" in validate_material(
        "x", {"name": "X", "category": "scm",
              "carbon": [{"value": 1.0, "source": "database"}],
              "strength_treatment": "magic"})


# --- R6.2: EPD plug-in + provenance ---------------------------------------------
EPD = {"epds": {"cement": {"value": 0.55, "unit": "kgCO2e/kg", "boundary": "A1-A3",
                           "reference": "EPD-BREVIK-2025"}}}


def test_validate_epd_json():
    assert validate_epd_json(EPD) is None
    assert validate_epd_json({"epds": {}}) is not None
    assert "unknown material" in validate_epd_json({"epds": {"unobtainium": {"value": 1}}})
    assert validate_epd_json({"epds": {"cement": {"value": "high"}}}) is not None
    assert validate_epd_json([1, 2]) is not None


def test_epd_overrides_registry_factor():
    factors = effective_carbon_factors(EPD["epds"])
    assert factors["cement"] == 0.55           # EPD beats the registry record
    assert factors["slag"] == 0.052            # everything else untouched


def test_provenance_resolution_order():
    factors = effective_carbon_factors(EPD["epds"])
    factors["slag"] = 0.9                       # user edited slag afterwards
    prov = carbon_provenance(factors, EPD["epds"])
    assert prov["cement"] == "epd:EPD-BREVIK-2025"
    assert prov["slag"] == "user-override"
    assert prov["ash"].startswith("database:")  # untouched registry default


# --- R6.4: pluggability — a JSON edit adds a material, no code change -----------
def test_new_material_via_registry_json(tmp_path):
    registry = json.loads(open("data/materials.json", encoding="utf-8").read())
    registry["materials"]["volcanic_ash"] = {
        "name": "Natural volcanic ash", "category": "Pozzolan",
        "dosage": {"default": 0, "max": 90, "unit": "kg/m3"},
        "unit_cost": {"value": 0.07, "currency": "USD/kg", "source": "test"},
        "carbon": [{"value": 0.006, "unit": "kgCO2e/kg", "boundary": "A1-A3",
                    "region": "test", "vintage": 2026, "source": "database",
                    "reference": "test", "uncertainty": 0.5}],
        "strength_treatment": "delta_estimate", "strength_factor": 0.03,
    }
    p = tmp_path / "materials.json"
    p.write_text(json.dumps(registry))
    try:
        set_materials_path(str(p))
        ex = exotics_view()
        assert "volcanic_ash" in ex
        assert ex["volcanic_ash"]["carbon_factor"] == 0.006
        assert ex["volcanic_ash"]["max"] == 90
        # Core views are untouched by the addition.
        assert carbon_factors_view() == EXPECTED_CARBON
        # And the exotic carbon/cost sums (what the UI computes) pick it up. The
        # module-level EXOTIC_ADMIXTURES snapshot is bound at import, so the live
        # view is the contract here; the app picks the JSON up on next start.
        dose = {k: 0 for k in ex}
        dose["volcanic_ash"] = 100
        carbon = sum(dose[k] * ex[k]["carbon_factor"] for k in ex)
        assert carbon == pytest.approx(0.6)
    finally:
        set_materials_path(None)
