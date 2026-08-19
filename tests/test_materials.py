"""R6 — material registry tests.

R6.1 gate: the legacy dicts, now views over data/materials.json, must be
BIT-IDENTICAL to the pre-registry literals (pinned below). R6.2: EPD plug-in and
provenance resolution. R6.4: a registry JSON edit (no code change) surfaces a new
material in every view — the pluggability contract.

R7.5 WP-2 gate: exotic admixtures carry density_kg_m3 too, surfaced through a
SEPARATE exotic_densities_view() so the pinned core-only densities_view() /
EXPECTED_DENSITIES below stays exactly seven entries (see
docs/specs/R7.5-chemistry-remediation.md WP-2).
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
    exotic_densities_view,
    exotics_view,
    slider_specs_view,
)

# --- R6.1: pinned pre-registry values (the migration must not move a digit) -----
EXPECTED_CARBON = {"cement": 0.912, "slag": 0.052, "ash": 0.004, "water": 0.0003,
                   "superplasticizer": 1.5, "coarse_agg": 0.008, "fine_agg": 0.005}
EXPECTED_COSTS = {"cement": 0.15, "slag": 0.08, "ash": 0.05, "water": 0.002,
                  "superplasticizer": 2.50, "coarse_agg": 0.03, "fine_agg": 0.04}
EXPECTED_DENSITIES = {"cement": 3150.0, "slag": 2900.0, "ash": 2300.0, "water": 1000.0,
                      "superplasticizer": 1100.0, "coarse_agg": 2700.0, "fine_agg": 2650.0}
# R7.4b: the mix-slider bounds, pinned to the pre-registry hardcoded literals in
# ui/state.py (the values SLIDER_SPECS carried before it became registry-driven).
EXPECTED_SLIDER_SPECS = {
    "cement": {"label": "Cement", "min": 100, "max": 550},
    "slag": {"label": "Slag", "min": 0, "max": 360},
    "ash": {"label": "Fly Ash", "min": 0, "max": 200},
    "water": {"label": "Water", "min": 120, "max": 250},
    "superplasticizer": {"label": "Superplasticizer", "min": 0, "max": 30},
    "coarse_agg": {"label": "Coarse Agg", "min": 700, "max": 1150},
    "fine_agg": {"label": "Fine Agg", "min": 550, "max": 1000},
}
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
# R7.5 WP-2: the twelve exotic densities, per the spec's table (WP-2 §Design item 1).
EXPECTED_EXOTIC_DENSITIES = {
    "silica_fume": 2200.0, "metakaolin": 2500.0, "rice_husk_ash": 2100.0,
    "limestone_filler": 2700.0, "calcined_clay": 2600.0, "steel_fiber": 7850.0,
    "polypropylene_fiber": 910.0, "basalt_fiber": 2650.0, "nano_silica": 2200.0,
    "graphene_oxide": 1800.0, "calcium_chloride": 2150.0, "shrink_reducer": 1000.0,
}


def test_views_are_bit_identical_to_pre_registry_literals():
    assert carbon_factors_view() == EXPECTED_CARBON
    assert unit_costs_view() == EXPECTED_COSTS
    assert densities_view() == EXPECTED_DENSITIES
    assert exotics_view() == EXPECTED_EXOTICS
    assert slider_specs_view() == EXPECTED_SLIDER_SPECS
    # Iteration ORDER matters too (the UI editors iterate these dicts).
    assert list(carbon_factors_view()) == list(EXPECTED_CARBON)
    assert list(exotics_view()) == list(EXPECTED_EXOTICS)


def test_exotic_densities_view_matches_spec_table_and_core_densities_untouched():
    """R7.5 WP-2: exotic_densities_view() is a SEPARATE view from densities_view() --
    it must carry all twelve exotic densities from the spec's table, while
    densities_view() / EXPECTED_DENSITIES above stays the pinned seven-entry core
    view, unwidened."""
    assert exotic_densities_view() == EXPECTED_EXOTIC_DENSITIES
    assert len(densities_view()) == 7
    assert set(exotic_densities_view()) & set(densities_view()) == set()


def test_legacy_module_dicts_are_the_views():
    from src.chemistry_simple import CARBON_FACTORS, UNIT_COSTS
    from src.physical import DENSITIES
    from src.exotics import EXOTIC_ADMIXTURES
    assert CARBON_FACTORS == EXPECTED_CARBON
    assert UNIT_COSTS == EXPECTED_COSTS
    assert DENSITIES == EXPECTED_DENSITIES
    assert EXOTIC_ADMIXTURES == EXPECTED_EXOTICS


def test_ui_slider_specs_are_registry_driven_in_param_order():
    """R7.4b: ui/state.py must no longer hardcode slider bounds -- it derives
    SLIDER_SPECS from slider_specs_view(), ordered by PARAM_NAMES ('age' is the one
    non-material entry, special-cased since it has no registry record)."""
    pytest.importorskip("streamlit")
    from src.generative_ga import PARAM_NAMES
    from ui.state import SLIDER_SPECS

    assert [p for p, *_ in SLIDER_SPECS] == list(PARAM_NAMES)
    for p, label, lo, hi in SLIDER_SPECS:
        if p == "age":
            assert (label, lo, hi) == ("Age (days)", 1, 365)
        else:
            spec = EXPECTED_SLIDER_SPECS[p]
            assert (label, lo, hi) == (spec["label"], spec["min"], spec["max"])


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


def test_validate_material_rejects_delta_estimate_missing_density():
    """R7.5 WP-2: an exotic (delta_estimate) record missing density_kg_m3 is rejected
    with a readable message -- the acceptance gate for the exotic volume-accounting
    fix (mix_volume can't be blind to a material's density if the registry refuses
    to load without one)."""
    rec = {
        "name": "X", "category": "Filler",
        "carbon": [{"value": 0.1, "source": "database"}],
        "strength_treatment": "delta_estimate",
        "dosage": {"default": 0, "max": 10, "unit": "kg/m3"},
        "strength_factor": 0.0,
        # density_kg_m3 deliberately omitted
    }
    err = validate_material("x", rec)
    assert err is not None
    assert "density_kg_m3" in err
    # Adding it makes the record valid.
    rec["density_kg_m3"] = 2000
    assert validate_material("x", rec) is None


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
        "density_kg_m3": 2400,
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


def test_slider_bound_edit_via_registry_json(tmp_path):
    """R7.4b pluggability: editing a core material's slider block in the JSON (no
    code change) changes the value slider_specs_view() reports -- the registry is
    the single UI authority for these bounds now, not a hardcoded ui/state.py list."""
    registry = json.loads(open("data/materials.json", encoding="utf-8").read())
    registry["materials"]["cement"]["slider"]["max"] = 600
    p = tmp_path / "materials.json"
    p.write_text(json.dumps(registry))
    try:
        set_materials_path(str(p))
        assert slider_specs_view()["cement"]["max"] == 600
        # Everything else is untouched.
        assert slider_specs_view()["slag"] == EXPECTED_SLIDER_SPECS["slag"]
    finally:
        set_materials_path(None)


def test_exotic_density_edit_via_registry_json(tmp_path):
    """R7.5 WP-2 acceptance gate: editing an exotic's density_kg_m3 in the JSON (no
    code change) changes the value exotic_densities_view() reports, and nothing
    else -- the registry is the single authority for exotic densities too."""
    registry = json.loads(open("data/materials.json", encoding="utf-8").read())
    registry["materials"]["calcined_clay"]["density_kg_m3"] = 2222
    p = tmp_path / "materials.json"
    p.write_text(json.dumps(registry))
    try:
        set_materials_path(str(p))
        assert exotic_densities_view()["calcined_clay"] == 2222.0
        # Everything else, including core densities, is untouched.
        rest = {k: v for k, v in EXPECTED_EXOTIC_DENSITIES.items() if k != "calcined_clay"}
        edited = {k: v for k, v in exotic_densities_view().items() if k != "calcined_clay"}
        assert edited == rest
        assert densities_view() == EXPECTED_DENSITIES
    finally:
        set_materials_path(None)
