"""
chemistry_simple.py - Tier 1: Linear Constituent Analysis

A simplified model for carbon emissions and material costs based on 
mass-weighted factors. This is NOT a true chemistry model—it treats 
concrete as a bag of inert components.

For molecular-level analysis, see chemistry_advanced.py.
"""
from typing import Dict

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

def calculate_embodied_carbon(mix: Dict[str, float], transport_km: float = 0.0,
                              factors: Dict[str, float] = None) -> float:
    """
    Calculates embodied carbon for a concrete mix (kg CO2 per m³).
    
    This is a LINEAR model: Carbon = Σ(mass_i × factor_i).
    It does NOT account for:
    - Clinker substitution ratios
    - Regional electricity grid carbon intensity
    - Hydration chemistry
    """
    factors = factors or CARBON_FACTORS
    carbon = sum(mix.get(k, 0) * factors.get(k, 0) for k in factors)

    # Transport heuristic: 0.1 kg CO2 per tonne per km. Sum only material masses
    # (the factor keys) -- 'age' is a curing time in days, not a mass.
    total_mass = sum(mix.get(k, 0) for k in factors)
    carbon += (total_mass / 1000.0) * transport_km * 0.1
    
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
