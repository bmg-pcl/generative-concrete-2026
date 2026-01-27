"""
chemistry_simple.py - Tier 1: Linear Constituent Analysis

A simplified model for carbon emissions and material costs based on 
mass-weighted factors. This is NOT a true chemistry model—it treats 
concrete as a bag of inert components.

For molecular-level analysis, see chemistry_advanced.py.
"""
import numpy as np
from typing import Dict

# Industry standard CO2 emission factors (kg CO2 / kg material)
# Values are approximate and can vary by region/producer
CARBON_FACTORS = {
    "cement": 0.912,        # High for Portland Cement (assumes ~95% clinker)
    "slag": 0.052,          # Low (byproduct of steel)
    "ash": 0.004,           # Very low (byproduct of coal)
    "water": 0.0003,        # Minimal
    "superplasticizer": 1.5, # High per kg, but used in small amounts
    "coarse_agg": 0.008,    # Quarrying and crushing
    "fine_agg": 0.005,      # Extraction and processing
}

# Approximate Unit Costs (Currency / kg)
UNIT_COSTS = {
    "cement": 0.15,
    "slag": 0.08,
    "ash": 0.05,
    "water": 0.002,
    "superplasticizer": 2.50,
    "coarse_agg": 0.03,
    "fine_agg": 0.04,
}

def calculate_mix_cost(mix: Dict[str, float], custom_costs: Dict[str, float] = None) -> float:
    """Calculates the total material cost per m³."""
    costs = custom_costs or UNIT_COSTS
    return sum(mix.get(k, 0) * costs.get(k, 0) for k in costs)

def calculate_embodied_carbon(mix: Dict[str, float], transport_km: float = 0.0) -> float:
    """
    Calculates embodied carbon for a concrete mix (kg CO2 per m³).
    
    This is a LINEAR model: Carbon = Σ(mass_i × factor_i).
    It does NOT account for:
    - Clinker substitution ratios
    - Regional electricity grid carbon intensity
    - Hydration chemistry
    """
    carbon = sum(mix.get(k, 0) * CARBON_FACTORS.get(k, 0) for k in CARBON_FACTORS)
    
    # Transport heuristic: 0.1 kg CO2 per tonne per km
    total_mass = sum(mix.values())
    carbon += (total_mass / 1000.0) * transport_km * 0.1
    
    return carbon

def estimate_curing_time(mix: Dict[str, float]) -> float:
    """
    Heuristic to estimate curing time to reach 70% strength (days).
    """
    w_c_ratio = mix.get("water", 180) / max(mix.get("cement", 300), 1)
    ash_slag_ratio = (mix.get("ash", 0) + mix.get("slag", 0)) / max(mix.get("cement", 300), 1)
    
    base_days = 7.0
    base_days += (w_c_ratio - 0.4) * 10
    base_days += ash_slag_ratio * 5
    
    return max(1.0, base_days)
