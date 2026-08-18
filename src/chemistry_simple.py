"""
chemistry_simple.py - Tier 1: Linear Constituent Analysis

A simplified model for carbon emissions and material costs based on
mass-weighted factors. This is NOT a true chemistry model—it treats
concrete as a bag of inert components.

For molecular-level analysis, see chemistry_advanced.py.

R7.5 WP-5: `transport_carbon` is the single definition of the transport heuristic,
shared by both tiers and by `ui_logic.carbon_breakdown` (previously the same
expression was triplicated across chemistry_simple.py, chemistry_advanced.py, and
ui_logic.py, and only summed the seven core factor keys -- exotic admixture mass
shipped for free). `exotic=None` reproduces the pre-WP-5 core-only mass sum exactly,
the same convention `physical.mix_volume(mix, exotic=None)` established in WP-2.
"""
from typing import Dict, Optional

from .materials import carbon_factors_view, unit_costs_view

# Cradle-to-gate CO2 emission factors (kg CO2e / kg material) and unit costs ($/kg).
# Since R6.1 these are VIEWS over the material registry (data/materials.json) — the
# values are unchanged (ICE database / WBCSD-CSI protocol defaults), but the registry
# record is the source of truth: it carries the provenance (boundary, region, vintage,
# source, reference) a bare scalar cannot. Editable per region in the app's Config
# tab; supplier EPDs plug in via src/materials.py (effective_carbon_factors).
CARBON_FACTORS = carbon_factors_view()
UNIT_COSTS = unit_costs_view()

def calculate_mix_cost(mix: Dict[str, float], custom_costs: Dict[str, float] = None) -> float:
    """Calculates the total material cost per m³."""
    costs = custom_costs or UNIT_COSTS
    return sum(mix.get(k, 0) * costs.get(k, 0) for k in costs)

def transport_carbon(mix: Dict[str, float], transport_km: float,
                     factors: Dict[str, float],
                     exotic: Optional[Dict[str, float]] = None) -> float:
    """
    Transport heuristic (kg CO2/m³): 0.1 kg CO2 per tonne per km, applied ONE-WAY
    to the distance entered (see ui/config.py's help text -- the factor is not
    doubled for a return leg).

    Sums mass over `factors`' keys (the core mix constituents with a carbon
    factor) -- 'age' is a curing time in days, not a mass, so it is excluded in
    both tiers. When `exotic` (a {material: kg/m3} dosing dict, e.g. from
    src/exotics.py) is given, its mass is added too, since exotic admixtures are
    physically hauled to site same as everything else. Default `exotic=None`
    reproduces the pre-WP-5 core-only mass sum exactly -- the single definition
    that both tiers and `ui_logic.carbon_breakdown` now share.
    """
    total_mass = sum(mix.get(k, 0) for k in factors)
    if exotic:
        total_mass += sum(exotic.values())
    return (total_mass / 1000.0) * transport_km * 0.1


def calculate_embodied_carbon(mix: Dict[str, float], transport_km: float = 0.0,
                              factors: Dict[str, float] = None,
                              exotic: Optional[Dict[str, float]] = None) -> float:
    """
    Calculates embodied carbon for a concrete mix (kg CO2 per m³).

    This is a LINEAR model: Carbon = Σ(mass_i × factor_i).
    It does NOT account for:
    - Clinker substitution ratios
    - Regional electricity grid carbon intensity
    - Hydration chemistry

    `exotic`, if given, is included in the transport mass (see `transport_carbon`);
    it does NOT add the exotic materials' own carbon factor -- that stays in
    `exotics.exotic_carbon`, which callers already add separately (see
    `ui_logic.compute_metrics`).
    """
    factors = factors or CARBON_FACTORS
    carbon = sum(mix.get(k, 0) * factors.get(k, 0) for k in factors)
    carbon += transport_carbon(mix, transport_km, factors, exotic=exotic)
    return carbon

def estimate_curing_time(mix: Dict[str, float]) -> float:
    """
    Heuristic to estimate curing time to reach 70% strength (days).

    Two effects, charged once each (R7.5 WP-3): dilution, via water/BINDER, and the
    genuinely slower reaction rate of SCMs, via a bounded SCM fraction. The previous
    version used water/CEMENT *and* an unbounded (ash+slag)/cement term, so replacing
    cement with slag was penalised twice for the same substitution — a 200 cement /
    200 slag / 180 water mix reported 17 days off an apparent w/c of 0.9, when its
    true w/b is 0.45.

    Still a heuristic, not a model; it just no longer contradicts itself.
    """
    cement = mix.get("cement", 300)
    slag = mix.get("slag", 0)
    ash = mix.get("ash", 0)
    water = mix.get("water", 180)

    binder = cement + slag + ash
    denom = max(binder, 1)                  # zero-binder mixes cannot divide by zero;
    wb_ratio = water / denom                # when binder == 0, slag == ash == 0 too,
    scm_fraction = (slag + ash) / denom     # so scm_fraction stays 0.

    base_days = 7.0
    base_days += (wb_ratio - 0.4) * 10
    base_days += scm_fraction * 5           # bounded in [0, 1] -> at most +5 days

    return max(1.0, base_days)
