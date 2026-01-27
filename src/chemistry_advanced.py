"""
chemistry_advanced.py - Tier 2: Molecular-Level Generative Chemistry

A thermodynamic and kinetic simulation layer for cement hydration.
Provides both FORWARD (analysis) and INVERSE (generative) modes.
"""
import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass

# ============================================================================
# OXIDE COMPOSITIONS (Typical values, should be loaded from JSON in production)
# ============================================================================
DEFAULT_OXIDE_COMPOSITIONS = {
    "OPC": {"CaO": 65.0, "SiO2": 21.0, "Al2O3": 5.5, "Fe2O3": 3.0, "SO3": 2.5, "MgO": 2.0},
    "SLAG": {"CaO": 40.0, "SiO2": 35.0, "Al2O3": 12.0, "Fe2O3": 0.5, "MgO": 8.0},
    "FLY_ASH_F": {"CaO": 5.0, "SiO2": 55.0, "Al2O3": 25.0, "Fe2O3": 8.0},
    "FLY_ASH_C": {"CaO": 20.0, "SiO2": 40.0, "Al2O3": 18.0, "Fe2O3": 6.0},
}

# ============================================================================
# BOGUE CALCULATION (Clinker Phase Estimation)
# ============================================================================
@dataclass
class ClinkerPhases:
    """Major clinker phases from Bogue calculation (wt%)."""
    C3S: float  # Alite
    C2S: float  # Belite
    C3A: float  # Tricalcium Aluminate
    C4AF: float # Ferrite

def bogue_calculation(oxides: Dict[str, float]) -> ClinkerPhases:
    """
    Classic Bogue calculation to estimate clinker phase composition.
    
    Assumes:
    - CaO, SiO2, Al2O3, Fe2O3 in wt%
    - Remainder is minor oxides (MgO, SO3, etc.)
    
    Reference: Bogue, R.H. (1929) "Calculation of the Compounds in Portland Cement"
    """
    CaO = oxides.get("CaO", 65.0)
    SiO2 = oxides.get("SiO2", 21.0)
    Al2O3 = oxides.get("Al2O3", 5.5)
    Fe2O3 = oxides.get("Fe2O3", 3.0)
    SO3 = oxides.get("SO3", 2.5)
    
    # Bogue equations
    C3S = 4.071 * CaO - 7.600 * SiO2 - 6.718 * Al2O3 - 1.430 * Fe2O3 - 2.852 * SO3
    C2S = 2.867 * SiO2 - 0.7544 * C3S
    C3A = 2.650 * Al2O3 - 1.692 * Fe2O3
    C4AF = 3.043 * Fe2O3
    
    # Clamp to reasonable values
    C3S = max(0, min(80, C3S))
    C2S = max(0, min(50, C2S))
    C3A = max(0, min(15, C3A))
    C4AF = max(0, min(20, C4AF))
    
    return ClinkerPhases(C3S=C3S, C2S=C2S, C3A=C3A, C4AF=C4AF)

# ============================================================================
# HYDRATION MODEL
# ============================================================================
@dataclass
class HydrationState:
    """State of cement hydration at a given time."""
    age_days: float
    degree_of_hydration: float  # 0 to 1
    CSH_content: float          # C-S-H gel (wt% of paste)
    CH_content: float           # Portlandite (Ca(OH)2)
    heat_released_kJ_kg: float

def hydration_kinetics(phases: ClinkerPhases, w_c_ratio: float, age_days: float) -> HydrationState:
    """
    Simplified Parrot & Killoh style hydration model.
    
    Models the degree of hydration based on:
    - Phase composition (C3S reacts fastest)
    - Water availability (w/c ratio)
    - Time (logarithmic approach to ultimate degree)
    """
    # Ultimate degree of hydration (limited by water)
    alpha_ult = min(1.0, w_c_ratio / 0.38)
    
    # Time-dependent hydration (Avrami-style)
    k = 0.15 * (phases.C3S / 60.0)  # Faster with more alite
    alpha = alpha_ult * (1 - np.exp(-k * age_days ** 0.6))
    
    # Phase products (simplified stoichiometry)
    CSH = alpha * (phases.C3S + phases.C2S) * 0.7
    CH = alpha * (phases.C3S * 0.3 + phases.C2S * 0.15)
    
    # Heat of hydration (J/g clinker phase)
    heat = alpha * (phases.C3S * 500 + phases.C2S * 250 + phases.C3A * 1340 + phases.C4AF * 420) / 1000.0
    
    return HydrationState(
        age_days=age_days,
        degree_of_hydration=alpha,
        CSH_content=CSH,
        CH_content=CH,
        heat_released_kJ_kg=heat
    )

# ============================================================================
# POZZOLANIC REACTION MODEL
# ============================================================================
def pozzolanic_reaction(
    CH_available: float, 
    pozzolan_mass: float, 
    pozzolan_type: str, 
    age_days: float
) -> Tuple[float, float]:
    """
    Models the pozzolanic reaction: CH + Pozzolan → C-S-H.
    
    Returns:
        (CH_consumed, additional_CSH)
    """
    # Reactivity factors (Class F ash is slower than slag)
    reactivity = {"FLY_ASH_F": 0.3, "FLY_ASH_C": 0.5, "SLAG": 0.8}
    k = reactivity.get(pozzolan_type, 0.4)
    
    # Reaction extent (time-dependent)
    extent = min(1.0, k * np.log1p(age_days / 7.0))
    
    # CH consumed proportional to pozzolan mass and silica content
    CH_consumed = min(CH_available, pozzolan_mass * 0.2 * extent)
    additional_CSH = CH_consumed * 1.5
    
    return CH_consumed, additional_CSH

# ============================================================================
# CARBON FROM CLINKER (More Accurate than Linear Model)
# ============================================================================
def carbon_from_clinker(
    cement_mass: float, 
    clinker_factor: float = 0.95,
    kiln_fuel_carbon: float = 0.35
) -> float:
    """
    Calculates CO2 emissions based on clinker chemistry.
    
    The main sources are:
    1. Calcination of limestone: CaCO3 → CaO + CO2 (~0.53 kg CO2/kg clinker)
    2. Fuel combustion in the kiln (~0.35 kg CO2/kg clinker, varies by fuel)
    
    Args:
        cement_mass: Mass of cement (kg/m³)
        clinker_factor: Fraction of cement that is clinker (e.g., 0.95 for OPC, 0.65 for LC3)
        kiln_fuel_carbon: Carbon intensity of kiln fuel
    """
    clinker_mass = cement_mass * clinker_factor
    calcination_co2 = clinker_mass * 0.53
    fuel_co2 = clinker_mass * kiln_fuel_carbon
    return calcination_co2 + fuel_co2

# ============================================================================
# INVERSE PLANNER (Generative Mode)
# ============================================================================
def inverse_plan_mix(
    target_strength_mpa: float,
    target_carbon_kg: float,
    max_cost: float
) -> Dict[str, float]:
    """
    Given target properties, generate a plausible mix design.
    
    This is a simplified heuristic solver. A full implementation would use
    constrained optimization or the BayesFlow amortizer.
    
    Returns:
        A dictionary of mix components (kg/m³).
    """
    # Heuristic: Higher strength → more cement, lower carbon → more SCMs
    base_cement = 300 + (target_strength_mpa - 30) * 8
    base_cement = max(200, min(550, base_cement))
    
    # Carbon constraint: reduce cement if carbon is tight
    if target_carbon_kg < base_cement * 0.9:
        scm_fraction = (base_cement * 0.9 - target_carbon_kg) / (base_cement * 0.9)
        slag = base_cement * min(0.5, scm_fraction)
        cement = base_cement - slag
    else:
        cement = base_cement
        slag = 0
    
    # Water based on w/c ratio
    w_c = 0.45 if target_strength_mpa < 40 else 0.35
    water = cement * w_c
    
    return {
        "cement": cement,
        "slag": slag,
        "ash": 0,
        "water": water,
        "superplasticizer": 5 if w_c < 0.4 else 0,
        "coarse_agg": 1000,
        "fine_agg": 750,
        "age": 28
    }

# ============================================================================
# ANALYSIS REPORT (Full Forward Pass)
# ============================================================================
def analyze_mix(mix: Dict[str, float], oxide_compositions: Dict = None) -> Dict:
    """
    Full molecular-level analysis of a concrete mix.
    
    Returns a comprehensive report including:
    - Clinker phases (Bogue)
    - Hydration state at 28 days
    - Carbon breakdown
    - Pozzolanic contribution
    """
    oxide_compositions = oxide_compositions or DEFAULT_OXIDE_COMPOSITIONS
    
    # Get clinker phases from OPC oxide composition
    phases = bogue_calculation(oxide_compositions["OPC"])
    
    # Hydration at 28 days
    w_c = mix.get("water", 180) / max(mix.get("cement", 300), 1)
    hydration = hydration_kinetics(phases, w_c, age_days=mix.get("age", 28))
    
    # Pozzolanic reaction (if slag or ash present)
    pozzolanic_csh = 0.0
    if mix.get("slag", 0) > 0:
        _, csh = pozzolanic_reaction(hydration.CH_content, mix["slag"], "SLAG", mix.get("age", 28))
        pozzolanic_csh += csh
    if mix.get("ash", 0) > 0:
        _, csh = pozzolanic_reaction(hydration.CH_content, mix["ash"], "FLY_ASH_F", mix.get("age", 28))
        pozzolanic_csh += csh
    
    # Carbon from clinker chemistry
    carbon = carbon_from_clinker(mix.get("cement", 300))
    
    return {
        "clinker_phases": phases,
        "hydration": hydration,
        "total_CSH": hydration.CSH_content + pozzolanic_csh,
        "carbon_kg_m3": carbon,
        "pozzolanic_CSH_contribution": pozzolanic_csh
    }

if __name__ == "__main__":
    # Example usage
    test_mix = {"cement": 350, "slag": 100, "ash": 0, "water": 160, "coarse_agg": 1000, "fine_agg": 750, "age": 28}
    report = analyze_mix(test_mix)
    
    print("=== Molecular Analysis Report ===")
    print(f"Clinker Phases: C3S={report['clinker_phases'].C3S:.1f}%, C2S={report['clinker_phases'].C2S:.1f}%")
    print(f"Degree of Hydration (28d): {report['hydration'].degree_of_hydration:.2%}")
    print(f"Total C-S-H (incl. pozzolanic): {report['total_CSH']:.1f} wt%")
    print(f"Clinker-based CO2: {report['carbon_kg_m3']:.1f} kg/m³")
