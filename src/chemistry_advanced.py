"""
chemistry_advanced.py - Tier 2: Molecular-Level Generative Chemistry

A thermodynamic and kinetic simulation layer for cement hydration.
Provides both FORWARD (analysis) and INVERSE (generative) modes.
"""
import os
import json
import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass

# Reuse the Tier-1 emission factors so the two tiers share the SAME non-cement
# carbon accounting (aggregates, water, SCMs, admixtures, transport). The advanced
# tier only differs by computing the CEMENT term from clinker chemistry.
from .chemistry_simple import CARBON_FACTORS as SIMPLE_CARBON_FACTORS

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_OXIDE_JSON = os.path.join(_DATA_DIR, "oxide_compositions.json")

# ============================================================================
# OXIDE COMPOSITIONS (Typical values, should be loaded from JSON in production)
# ============================================================================
DEFAULT_OXIDE_COMPOSITIONS = {
    "OPC": {"CaO": 65.0, "SiO2": 21.0, "Al2O3": 5.5, "Fe2O3": 3.0, "SO3": 2.5, "MgO": 2.0},
    "SLAG": {"CaO": 40.0, "SiO2": 35.0, "Al2O3": 12.0, "Fe2O3": 0.5, "MgO": 8.0},
    "FLY_ASH_F": {"CaO": 5.0, "SiO2": 55.0, "Al2O3": 25.0, "Fe2O3": 8.0},
    "FLY_ASH_C": {"CaO": 20.0, "SiO2": 40.0, "Al2O3": 18.0, "Fe2O3": 6.0},
}


def load_oxide_compositions() -> Dict[str, Dict]:
    """Load cement/SCM oxide compositions (incl. clinker_factor) from data JSON.

    Falls back to DEFAULT_OXIDE_COMPOSITIONS if the file is missing or unreadable.
    """
    try:
        with open(_OXIDE_JSON) as f:
            return json.load(f)
    except (OSError, ValueError):
        return DEFAULT_OXIDE_COMPOSITIONS


def clinker_factor_for(cement_type: str = "OPC", oxide_compositions: Dict = None) -> float:
    """Clinker fraction for a cement type, read from the oxide JSON (OPC≈0.95,
    LC3≈0.50). Defaults to 0.95 when the type or field is absent."""
    comps = oxide_compositions or load_oxide_compositions()
    return float(comps.get(cement_type, {}).get("clinker_factor", 0.95))

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
    clinker_factor: float = None,
    kiln_fuel_carbon: float = 0.35,
    cement_type: str = "OPC",
) -> float:
    """
    Calculates CO2 emissions from the CEMENT term based on clinker chemistry.

    The main sources are:
    1. Calcination of limestone: CaCO3 → CaO + CO2 (~0.53 kg CO2/kg clinker)
    2. Fuel combustion in the kiln (~0.35 kg CO2/kg clinker, varies by fuel)

    Note: this is the cement-only contribution. For a full mix carbon figure on the
    same system boundary as the Tier-1 model, use `embodied_carbon_advanced`.

    Args:
        cement_mass: Mass of cement (kg/m³)
        clinker_factor: Fraction of cement that is clinker. If None, it is read from
            data/oxide_compositions.json for `cement_type` (e.g., 0.95 OPC, 0.50 LC3).
        kiln_fuel_carbon: Carbon intensity of kiln fuel
        cement_type: Key into the oxide JSON used when clinker_factor is None.
    """
    if clinker_factor is None:
        clinker_factor = clinker_factor_for(cement_type)
    clinker_mass = cement_mass * clinker_factor
    calcination_co2 = clinker_mass * 0.53
    fuel_co2 = clinker_mass * kiln_fuel_carbon
    return calcination_co2 + fuel_co2


def embodied_carbon_advanced(
    mix: Dict[str, float],
    transport_km: float = 0.0,
    cement_type: str = "OPC",
    factors: Dict[str, float] = None,
) -> float:
    """
    Full-mix embodied carbon (kg CO2/m³) on the SAME system boundary as the Tier-1
    `chemistry_simple.calculate_embodied_carbon`, but with the cement term replaced
    by clinker chemistry.

    carbon = clinker_chemistry(cement)                       # higher-fidelity cement
           + Σ mass_i · factor_i   for i ≠ cement            # shared non-cement terms
           + transport heuristic                             # shared

    The tier toggle therefore changes *fidelity*, not *scope*: switching to Advanced
    only refines how the cement contribution is computed, so the two tiers are
    directly comparable.
    """
    factors = factors or SIMPLE_CARBON_FACTORS
    carbon = carbon_from_clinker(mix.get("cement", 0.0), cement_type=cement_type)

    # Non-cement constituents, using the same (editable) Tier-1 factors.
    for component, factor in factors.items():
        if component == "cement":
            continue
        carbon += mix.get(component, 0.0) * factor

    # Same transport heuristic as Tier-1 (0.1 kg CO2 / tonne / km). Only material
    # masses count -- 'age' is a curing time, not a mass.
    total_mass = sum(mix.get(k, 0.0) for k in factors)
    carbon += (total_mass / 1000.0) * transport_km * 0.1

    return carbon

# ============================================================================
# INVERSE PLANNER (Generative Mode)
# ============================================================================
def inverse_plan_mix(
    target_strength_mpa: float,
    target_carbon_kg: float = None,
    max_cost: float = None
) -> Dict[str, float]:
    """
    Given target properties, generate a plausible mix design.

    This now delegates to the GA-based `PopulationInverseDesigner` (see
    src/generative_ga.py). It searches -- within the training-data envelope -- for
    a mix whose model-predicted strength matches the target, optionally penalising
    mixes above the carbon budget. This replaces the previous open-loop heuristic,
    which did not track the target (a 25 MPa request produced a ~51 MPa mix) and
    emitted out-of-distribution water below the dataset minimum.

    Args:
        target_strength_mpa: Desired compressive strength (MPa).
        target_carbon_kg: Optional soft carbon budget (kg CO2/m³).
        max_cost: Accepted for backward compatibility; not yet used as a hard
            constraint (a cost-aware objective is future work).

    Returns:
        A dictionary of mix components (kg/m³) plus age (days).
    """
    # Imported lazily so `chemistry_advanced` stays importable without the model
    # stack, and to avoid a module-load-time dependency cycle.
    from .generative_ga import PopulationInverseDesigner

    designer = PopulationInverseDesigner()
    return designer.best_mix(target_strength_mpa, carbon_target=target_carbon_kg)

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
