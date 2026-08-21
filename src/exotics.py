"""
exotics.py - Exotic admixtures: cost, carbon, and an OPT-IN strength model.

The XGBoost strength predictor is trained on the UCI dataset, which contains NONE
of these modern admixtures (silica fume, fibers, nano-silica, ...). So by default
they must NOT move the predicted strength -- only cost and carbon, which are simple
mass-factor sums that are valid for any material.

`exotic_strength_delta` is the config switch: it returns 0 unless explicitly
enabled. When enabled it applies a transparent linear estimate,

        delta_strength = Σ amount_i · strength_factor_i     (MPa)

The `strength_factor` values below are UNVALIDATED, order-of-magnitude placeholders
seeded from the general literature direction (pozzolans/nano densify and strengthen;
inert fillers slightly dilute; fibers do little for *compressive* strength). They
exist so the switch is demonstrable and so there is one obvious place to drop in
real, fitted coefficients once exotic strength data is collected. Until then, the
UI labels any exotic strength contribution as an unvalidated estimate.
"""
from typing import Dict, List

from .materials import exotics_view, load_materials

# Per-admixture properties (a VIEW over data/materials.json since R6.1 -- add a new
# admixture by appending a registry record, not by editing code; see
# docs/specs/R6-materials-platform.md section 2.2).
#   default / max : slider bounds (kg/m3), dosage typical of each material's scale
#   carbon_factor : kg CO2 per kg (valid mass factor)
#   cost          : $ per kg
#   category      : grouping for the UI
#   strength_factor: MPa per kg -- PLACEHOLDER estimate, only used when the exotic
#                    strength switch is ON. Replace with fitted values when data exists.
EXOTIC_ADMIXTURES: Dict[str, Dict] = exotics_view()

# Shown in the UI wherever an exotic strength estimate is applied.
EXOTIC_STRENGTH_DISCLAIMER = (
    "Exotic strength contribution is an UNVALIDATED linear estimate (placeholder "
    "coefficients), applied only because the experimental switch is enabled. Replace "
    "with fitted coefficients once real exotic strength data is available."
)


def exotic_carbon(exotic: Dict[str, float]) -> float:
    """Extra embodied carbon (kg CO2/m3) from exotic admixtures. Always valid."""
    return sum(exotic.get(k, 0) * props["carbon_factor"] for k, props in EXOTIC_ADMIXTURES.items())


def exotic_cost(exotic: Dict[str, float]) -> float:
    """Extra material cost ($/m3) from exotic admixtures. Always valid."""
    return sum(exotic.get(k, 0) * props["cost"] for k, props in EXOTIC_ADMIXTURES.items())


def exotic_strength_delta(exotic: Dict[str, float], enabled: bool = False) -> float:
    """
    Strength contribution (MPa) from exotic admixtures.

    Returns 0.0 when `enabled` is False (the default) -- the strength model does not
    know about exotics, so they must not silently move the prediction. When True,
    applies the placeholder linear estimate documented above.
    """
    if not enabled:
        return 0.0
    return sum(exotic.get(k, 0) * props["strength_factor"] for k, props in EXOTIC_ADMIXTURES.items())


def compliance_warnings(exotic: Dict[str, float]) -> List[str]:
    """
    Advisory compliance warnings (R8.0 WP-D2) -- one string per DOSED material
    that carries a registry `"restrictions"` block (e.g. calcium_chloride's
    ACI 318 ch. 19 / EN 206 chloride limit in reinforced concrete). Styled after
    `physical.workability_flag`: advisory, not a hard constraint -- this tool
    does not know whether the mix is going into reinforced concrete, it only
    flags that the question exists. `{}` or nothing dosed returns [].

    This is the data hook R8.2's compliance layer will consume; the UI wiring
    (Compare tab / ticket) is Wave B.
    """
    registry = load_materials()
    warnings = []
    for k, amount in (exotic or {}).items():
        if not amount:
            continue
        restrictions = registry.get(k, {}).get("restrictions")
        if not restrictions:
            continue
        detail = ", ".join(f"{ctx}: {status}" for ctx, status in restrictions.items()
                           if ctx != "reference")
        reference = restrictions.get("reference", "unreferenced")
        warnings.append(f"{k}: {detail} (see {reference}).")
    return warnings
