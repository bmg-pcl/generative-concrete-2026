"""materials.py — the single material registry (spec R6).

Every material the tool knows — core mix constituents and exotic admixtures — is one
declarative record in data/materials.json: identity, density, dosage bounds, cost,
provenance-tagged carbon records, chemistry hooks, and how the strength model treats
it. The legacy dicts (CARBON_FACTORS, UNIT_COSTS, DENSITIES, EXOTIC_ADMIXTURES) are
views over this registry, so adding a material is a JSON edit, not a code change
(see docs/specs/R6-materials-platform.md §2.2 for the recipe).

Carbon values are records, never bare scalars: value, unit, boundary, region, vintage,
source (epd | database | parametric | placeholder), reference, uncertainty. Resolution
order for the effective factor (R6.2): project override > supplier EPD > registry
record > placeholder — and `carbon_provenance` reports which one won, per material,
for the mix ticket.
"""
import json
import math
import os
from typing import Dict, Optional

DEFAULT_MATERIALS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "materials.json"
)
# Overridable for tests and for plugging in an alternative registry without code
# changes (env var read at import; set_materials_path wins at runtime).
_materials_path = os.environ.get("MATERIALS_JSON", DEFAULT_MATERIALS_PATH)
_cache: Dict[str, dict] = {}

STRENGTH_TREATMENTS = ("trained_feature", "delta_estimate", "inert")
CARBON_SOURCES = ("epd", "database", "parametric", "placeholder")


def set_materials_path(path: Optional[str]):
    """Point the registry at a different JSON (None resets to the default)."""
    global _materials_path
    _materials_path = path or os.environ.get("MATERIALS_JSON", DEFAULT_MATERIALS_PATH)
    _cache.clear()


def load_materials(path: Optional[str] = None) -> Dict[str, dict]:
    """The full registry {key: record}, cached per path."""
    p = path or _materials_path
    if p not in _cache:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        materials = data["materials"]
        for key, rec in materials.items():
            err = validate_material(key, rec)
            if err:
                raise ValueError(f"materials registry: {err}")
        _cache[p] = materials
    return _cache[p]


def validate_material(key: str, rec: dict) -> Optional[str]:
    """Return None if a record is usable, else a human-readable error."""
    for field in ("name", "category", "carbon", "strength_treatment"):
        if field not in rec:
            return f"'{key}' is missing '{field}'"
    if rec["strength_treatment"] not in STRENGTH_TREATMENTS:
        return f"'{key}' has unknown strength_treatment '{rec['strength_treatment']}'"
    if not rec["carbon"]:
        return f"'{key}' must carry at least one carbon record (never a bare scalar)"
    for c in rec["carbon"]:
        if "value" not in c or not isinstance(c["value"], (int, float)):
            return f"'{key}' has a carbon record without a numeric 'value'"
        if c.get("source") not in CARBON_SOURCES:
            return f"'{key}' carbon record needs source in {CARBON_SOURCES}"
    if rec["strength_treatment"] == "trained_feature":
        if "density_kg_m3" not in rec:
            return f"'{key}' is a trained feature and must declare density_kg_m3"
        if "unit_cost" not in rec:
            return f"'{key}' is a trained feature and must declare unit_cost"
        if "slider" not in rec or not {"label", "min", "max"} <= set(rec["slider"]):
            return f"'{key}' is a trained feature and must declare slider {{label, min, max}}"
    if rec["strength_treatment"] == "delta_estimate":
        if "dosage" not in rec or "strength_factor" not in rec:
            return f"'{key}' is a delta_estimate material and must declare dosage and strength_factor"
        if "density_kg_m3" not in rec:
            return f"'{key}' is a delta_estimate material and must declare density_kg_m3"
    return None


def get_material(key: str) -> dict:
    return load_materials()[key]


def all_materials(category: Optional[str] = None) -> Dict[str, dict]:
    mats = load_materials()
    if category is None:
        return dict(mats)
    return {k: v for k, v in mats.items() if v["category"] == category}


def _core_items():
    return [(k, v) for k, v in load_materials().items()
            if v["strength_treatment"] == "trained_feature"]


# -- legacy views (the four dicts the rest of the code was built on) ----------
def carbon_factors_view() -> Dict[str, float]:
    """{material: A1-A3 kgCO2e/kg} for the core mix constituents."""
    return {k: float(v["carbon"][0]["value"]) for k, v in _core_items()}


def unit_costs_view() -> Dict[str, float]:
    return {k: float(v["unit_cost"]["value"]) for k, v in _core_items()}


def densities_view() -> Dict[str, float]:
    return {k: float(v["density_kg_m3"]) for k, v in _core_items()}


def exotic_densities_view() -> Dict[str, float]:
    """{material: density kg/m3} for the exotic (delta_estimate) admixtures.

    Deliberately separate from densities_view() (R7.5 WP-2): that view is pinned
    core-only (EXPECTED_DENSITIES in tests/test_materials.py, an R6.1 bit-identical
    migration gate) and must stay exactly seven entries, so exotic densities get
    their own view rather than widening it. Consumed by physical.mix_volume's
    optional `exotic` argument."""
    return {k: float(v["density_kg_m3"]) for k, v in load_materials().items()
            if v["strength_treatment"] == "delta_estimate"}


def slider_specs_view() -> Dict[str, dict]:
    """{material: {"label": ..., "min": ..., "max": ...}} for the core mix sliders.

    Deliberately does not know PARAM_NAMES or slider ORDER — src/generative_ga.py
    (which owns PARAM_NAMES) sits above chemistry_simple.py, which imports from this
    module, so importing PARAM_NAMES here would be circular. Callers that need a
    specific order (ui/state.py) look each material up by key instead."""
    return {k: dict(v["slider"]) for k, v in _core_items()}


def exotics_view() -> Dict[str, dict]:
    """The EXOTIC_ADMIXTURES shape, derived from the registry."""
    out = {}
    for k, v in load_materials().items():
        if v["strength_treatment"] == "trained_feature":
            continue
        out[k] = {
            "default": v["dosage"]["default"], "max": v["dosage"]["max"],
            "carbon_factor": float(v["carbon"][0]["value"]),
            "cost": float(v["unit_cost"]["value"]),
            "category": v["category"],
            "strength_factor": float(v.get("strength_factor", 0.0)),
        }
    return out


# -- carbon provenance & EPD plug-in (R6.2) -----------------------------------
def validate_epd_json(data) -> Optional[str]:
    """Return None if an uploaded EPD file is usable, else an error message.

    Schema: {"epds": {"<material>": {"value": <kgCO2e/kg>, "reference": "...",
    "boundary": "A1-A3", ...}}} — unknown materials are rejected so a typo cannot
    silently attach to nothing."""
    if not isinstance(data, dict) or not isinstance(data.get("epds"), dict):
        return 'EPD file must be a JSON object with an "epds" mapping.'
    if not data["epds"]:
        return "EPD file contains no entries."
    known = set(load_materials().keys())
    for mat, rec in data["epds"].items():
        if mat not in known:
            return f"EPD references unknown material '{mat}'."
        if not isinstance(rec, dict) or not isinstance(rec.get("value"), (int, float)):
            return f"EPD for '{mat}' needs a numeric 'value' (kgCO2e/kg)."
    return None


def effective_carbon_factors(epds: Optional[dict] = None) -> Dict[str, float]:
    """Registry factors with supplier EPDs applied (EPD beats registry record)."""
    factors = carbon_factors_view()
    for mat, rec in (epds or {}).items():
        if mat in factors:
            factors[mat] = float(rec["value"])
    return factors


def carbon_provenance(current_factors: Dict[str, float],
                      epds: Optional[dict] = None) -> Dict[str, str]:
    """Which source produced each *currently effective* factor.

    Resolution reported per material: a value equal to an attached EPD is that EPD;
    equal to the registry record is that record's source/reference; anything else is
    a user override (rule 1 — the user edited the factor in Config or the config
    file). Printed on the mix ticket so a reader knows what each number rests on."""
    registry = load_materials()
    out = {}
    for mat, value in current_factors.items():
        epd = (epds or {}).get(mat)
        if epd is not None and math.isclose(float(value), float(epd["value"]), rel_tol=1e-9):
            out[mat] = f"epd:{epd.get('reference', 'unreferenced')}"
            continue
        rec = registry.get(mat, {}).get("carbon", [{}])[0]
        if "value" in rec and math.isclose(float(value), float(rec["value"]), rel_tol=1e-9):
            out[mat] = f"{rec.get('source', 'database')}:{rec.get('reference', '?')}"
        else:
            out[mat] = "user-override"
    return out
