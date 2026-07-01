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

# Per-admixture properties.
#   default / max : slider bounds (kg/m3), dosage typical of each material's scale
#   carbon_factor : kg CO2 per kg (valid mass factor)
#   cost          : $ per kg
#   category      : grouping for the UI
#   strength_factor: MPa per kg -- PLACEHOLDER estimate, only used when the exotic
#                    strength switch is ON. Replace with fitted values when data exists.
EXOTIC_ADMIXTURES: Dict[str, Dict] = {
    "silica_fume":         {"default": 0, "max": 50,  "carbon_factor": 0.02, "cost": 0.80,   "category": "Pozzolan", "strength_factor": 0.08},
    "metakaolin":          {"default": 0, "max": 80,  "carbon_factor": 0.30, "cost": 0.45,   "category": "Pozzolan", "strength_factor": 0.06},
    "rice_husk_ash":       {"default": 0, "max": 60,  "carbon_factor": 0.01, "cost": 0.15,   "category": "Pozzolan", "strength_factor": 0.04},
    "limestone_filler":    {"default": 0, "max": 100, "carbon_factor": 0.01, "cost": 0.05,   "category": "Filler",   "strength_factor": -0.01},
    "calcined_clay":       {"default": 0, "max": 150, "carbon_factor": 0.25, "cost": 0.12,   "category": "Filler",   "strength_factor": 0.03},
    "steel_fiber":         {"default": 0, "max": 80,  "carbon_factor": 1.80, "cost": 1.50,   "category": "Fiber",    "strength_factor": 0.05},
    "polypropylene_fiber": {"default": 0, "max": 10,  "carbon_factor": 3.50, "cost": 4.00,   "category": "Fiber",    "strength_factor": -0.02},
    "basalt_fiber":        {"default": 0, "max": 20,  "carbon_factor": 0.60, "cost": 2.50,   "category": "Fiber",    "strength_factor": 0.02},
    "nano_silica":         {"default": 0, "max": 5,   "carbon_factor": 5.00, "cost": 25.00,  "category": "Nano",     "strength_factor": 1.50},
    "graphene_oxide":      {"default": 0, "max": 1,   "carbon_factor": 50.00,"cost": 500.00, "category": "Nano",     "strength_factor": 8.00},
    "calcium_chloride":    {"default": 0, "max": 10,  "carbon_factor": 0.80, "cost": 0.30,   "category": "Chemical", "strength_factor": 0.10},
    "shrink_reducer":      {"default": 0, "max": 8,   "carbon_factor": 2.00, "cost": 6.00,   "category": "Chemical", "strength_factor": 0.00},
}

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
