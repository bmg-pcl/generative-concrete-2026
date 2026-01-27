import numpy as np
from typing import Dict

# Industry standard CO2 emission factors (kg CO2 / kg material)
# Values are approximate and can vary by region/producer
CARBON_FACTORS = {
    "cement": 0.912,        # High for Portland Cement
    "slag": 0.052,          # Low (byproduct)
    "ash": 0.004,           # Very low (byproduct)
    "water": 0.0003,        # Minimal
    "superplasticizer": 1.5, # High per kg, but used in small amounts
    "coarse_agg": 0.008,    # Quarrying and crushing
    "fine_agg": 0.005,      # Extraction and processing
}

# Approximate Unit Costs (Currency / kg)
# These are defaults and should be adjustable in the UI
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
    """Calculates the total material cost per m3."""
    costs = custom_costs or UNIT_COSTS
    total_cost = 0.0
    for component, amount in mix.items():
        if component in costs:
            total_cost += amount * costs[component]
    return total_cost
    """
    Calculates embodied carbon for a concrete mix (kg CO2 per m3).
    
    Args:
        mix: Dictionary with keys corresponding to UCI dataset columns.
        transport_km: Additional transport distance factor.
    """
    carbon = 0.0
    for component, amount in mix.items():
        if component in CARBON_FACTORS:
            carbon += amount * CARBON_FACTORS[component]
    
    # Simple transport heuristic: 0.1 kg CO2 per tonne per km (rough average for truck)
    # Total mass in kg
    total_mass = sum(mix.values())
    carbon += (total_mass / 1000.0) * transport_km * 0.1
    
    return carbon

def estimate_curing_time(mix: Dict[str, float]) -> float:
    """
    Heuristic to estimate curing time to reach 70% strength (days).
    Higher water/cement ratio generally means slower curing.
    Higher ash/slag often retards early strength.
    """
    w_c_ratio = mix.get("water", 180) / max(mix.get("cement", 300), 1)
    ash_slag_ratio = (mix.get("ash", 0) + mix.get("slag", 0)) / max(mix.get("cement", 300), 1)
    
    # Base days = 7. Adjust based on ratios.
    base_days = 7.0
    
    # Higher w/c ratio -> longer curing
    base_days += (w_c_ratio - 0.4) * 10
    
    # Higher replacement ratio -> longer curing at early age
    base_days += ash_slag_ratio * 5
    
    return max(1.0, base_days)
