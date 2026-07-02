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
from typing import Dict

from .materials import exotics_view

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
