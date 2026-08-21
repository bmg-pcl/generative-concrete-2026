"""
chemistry_advanced.py - Tier 2: Molecular-Level Generative Chemistry

A thermodynamic and kinetic simulation layer for cement hydration.
Provides both FORWARD (analysis) and INVERSE (generative) modes.

R7.5 / WP-1 status (see docs/specs/R7.5-chemistry-remediation.md): the Bogue ->
hydration -> pozzolanic chain below is DIMENSIONALLY CHECKED but UNCALIBRATED.
Units are now internally consistent (kg/m3 throughout the hydration chain, wt%
only for the raw Bogue clinker-phase output) and the portlandite budget is
conserved, but there is no calorimetry or degree-of-hydration data anywhere in
this repo to fit the reaction-rate or stoichiometry coefficients against. Treat
every number from `bogue_calculation` onward as a labelled, order-of-magnitude
sketch -- NOT a validated prediction. This is the opposite situation from the
carbon-accounting functions below (`embodied_carbon_advanced`,
`carbon_from_clinker`, `clinker_scope_split`), which ARE measured and tested
against reference values (WBCSD/CSI defaults, kiln energy benchmarks).

R8.0 / WP-B status (see docs/specs/R8.0-carbon-chemistry-deepening.md): the
carbon section was frozen for R7.5/WP-1 but WP-B is the one package licensed to
revise it deliberately -- it closes two scope holes (process electricity,
LC3's non-clinker constituent carbon) that were making the Tier-2 cement term
under-count relative to Tier-1 and to published LC3 EPDs. Every changed pin is
re-derived and re-justified in tests/test_chemistry.py, tests/test_clinker_source.py,
and the commit message. `clinker_scope_split` and its per-kg-clinker worked-
example pins are untouched by WP-B -- the new terms live in `carbon_from_clinker`,
per kg CEMENT.
"""
import json
import os
from dataclasses import dataclass

import numpy as np

# Reuse the Tier-1 emission factors so the two tiers share the SAME non-cement
# carbon accounting (aggregates, water, SCMs, admixtures, transport). The advanced
# tier only differs by computing the CEMENT term from clinker chemistry.
# `transport_carbon` (R7.5 WP-5) is the single shared definition of the transport
# heuristic -- see its docstring in chemistry_simple.py.
from .chemistry_simple import CARBON_FACTORS as SIMPLE_CARBON_FACTORS, transport_carbon
from .materials import get_material

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_OXIDE_JSON = os.path.join(_DATA_DIR, "oxide_compositions.json")

# ============================================================================
# OXIDE COMPOSITIONS (Typical values, should be loaded from JSON in production)
# ============================================================================
# `bogue_valid` mirrors data/oxide_compositions.json (R7.5 WP-1 finding 5): Bogue
# applies to Portland CLINKER oxides only. True for OPC; false for anything that
# is a blend or a byproduct oxide set that Bogue was never derived for.
DEFAULT_OXIDE_COMPOSITIONS = {
    "OPC": {"CaO": 65.0, "SiO2": 21.0, "Al2O3": 5.5, "Fe2O3": 3.0, "SO3": 2.5, "MgO": 2.0,
            "bogue_valid": True},
    "SLAG": {"CaO": 40.0, "SiO2": 35.0, "Al2O3": 12.0, "Fe2O3": 0.5, "MgO": 8.0,
              "bogue_valid": False},
    "FLY_ASH_F": {"CaO": 5.0, "SiO2": 55.0, "Al2O3": 25.0, "Fe2O3": 8.0,
                  "bogue_valid": False},
    "FLY_ASH_C": {"CaO": 20.0, "SiO2": 40.0, "Al2O3": 18.0, "Fe2O3": 6.0,
                  "bogue_valid": False},
}


def load_oxide_compositions() -> dict[str, dict]:
    """Load cement/SCM oxide compositions (incl. clinker_factor) from data JSON.

    Falls back to DEFAULT_OXIDE_COMPOSITIONS if the file is missing or unreadable.
    """
    try:
        with open(_OXIDE_JSON) as f:
            return json.load(f)
    except (OSError, ValueError):
        return DEFAULT_OXIDE_COMPOSITIONS


def clinker_factor_for(cement_type: str = "OPC", oxide_compositions: dict | None = None) -> float:
    """Clinker fraction for a cement type, read from the oxide JSON (OPC≈0.95,
    LC3≈0.50). Defaults to 0.95 when the type or field is absent."""
    comps = oxide_compositions or load_oxide_compositions()
    return float(comps.get(cement_type, {}).get("clinker_factor", 0.95))

# ============================================================================
# BOGUE CALCULATION (Clinker Phase Estimation)
# ============================================================================
@dataclass
class ClinkerPhases:
    """Major clinker phases from Bogue calculation (wt% of clinker)."""
    C3S: float  # Alite
    C2S: float  # Belite
    C3A: float  # Tricalcium Aluminate
    C4AF: float # Ferrite

def bogue_calculation(oxides: dict[str, float]) -> ClinkerPhases:
    """
    Classic Bogue calculation to estimate clinker phase composition.

    Assumes:
    - CaO, SiO2, Al2O3, Fe2O3 in wt%
    - Remainder is minor oxides (MgO, SO3, etc.) -- do not force the four major
      phases to sum to 100%; that remainder is real, not a rounding error.

    Bogue applies to Portland CLINKER oxides only (R7.5 WP-1 finding 5). A record
    carrying `"bogue_valid": false` (blended cements: SLAG, FLY_ASH_*, LC3 -- their
    oxide sets are not a clinker composition Bogue was derived for) raises instead
    of silently fabricating phase percentages. A record with no `bogue_valid` key
    at all (e.g. a bare oxide dict passed directly, not from the registry) is
    treated as valid for backward compatibility -- the flag is an explicit "no",
    not a required "yes".

    R8.0/WP-B B3 -- bare-dict hygiene: `CaO` and `SiO2` are the two oxides that
    dominate the Bogue equations and define what "OPC-like" even means, so a bare
    dict missing either raises `ValueError` naming the missing oxide(s) rather than
    silently impersonating a real OPC composition (the pre-WP-B behaviour: an empty
    dict returned full OPC-typical phases). `Al2O3`, `Fe2O3`, `SO3` are minor oxides
    with a real, if smaller, effect on the equations below; they keep their
    OPC-typical defaults for backward compatibility with partial records that DO
    carry the majors. Every registry record (data/oxide_compositions.json) carries
    all five oxides, so the registry-record path through `analyze_mix` is
    unaffected -- this only tightens the bare-dict path.

    Reference: Bogue, R.H. (1929) "Calculation of the Compounds in Portland Cement"
    """
    if oxides.get("bogue_valid", True) is False:
        raise ValueError(
            "bogue_calculation: this oxide record is flagged bogue_valid=false "
            "(not a Portland clinker composition -- Bogue does not apply to "
            "blended/SCM oxide sets). See data/oxide_compositions.json."
        )

    missing = [name for name in ("CaO", "SiO2") if name not in oxides]
    if missing:
        raise ValueError(
            "bogue_calculation: missing required major oxide(s) "
            f"{', '.join(missing)} -- a bare oxide dict must at least specify CaO "
            "and SiO2 (Bogue's dominant terms); defaults are only applied to the "
            "minor oxides (Al2O3, Fe2O3, SO3)."
        )

    CaO = oxides["CaO"]
    SiO2 = oxides["SiO2"]
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
# All state below the raw Bogue wt% is expressed in kg/m3 of concrete, or in
# kJ per kg of CEMENT for heat -- one unit system, named on every field
# (R7.5 WP-1 finding 2). The conversion from Bogue's wt%-of-clinker to kg/m3
# is `phase_pct / 100 * cement_mass_kg_m3 * clinker_factor`.
@dataclass
class HydrationState:
    """State of cement hydration at a given time. UNCALIBRATED (see module docstring)."""
    age_days: float
    degree_of_hydration: float  # 0 to 1, dimensionless
    CSH_kg_m3: float            # C-S-H gel formed from clinker phases, kg per m3 of concrete
    CH_kg_m3: float             # Portlandite (Ca(OH)2) produced, kg per m3 of concrete
    heat_kJ_per_kg_cement: float  # cumulative heat released, kJ per kg of CEMENT (not clinker)

def hydration_kinetics(
    phases: ClinkerPhases,
    w_c_ratio: float,
    age_days: float,
    cement_mass_kg_m3: float,
    clinker_factor: float,
) -> HydrationState:
    """
    Simplified Parrot & Killoh style hydration model.

    Models the degree of hydration based on:
    - Phase composition (C3S reacts fastest)
    - Water availability (w/c ratio)
    - Time (logarithmic approach to ultimate degree)

    Args:
        phases: Bogue clinker phases (wt% of CLINKER).
        w_c_ratio: water/cement mass ratio.
        age_days: curing age.
        cement_mass_kg_m3: cement dosage of the mix, kg per m3 of concrete.
        clinker_factor: fraction of the cement that is clinker (OPC≈0.95, LC3≈0.50).
    """
    # Ultimate degree of hydration (limited by water)
    alpha_ult = min(1.0, w_c_ratio / 0.38)

    # Time-dependent hydration (Avrami-style)
    k = 0.15 * (phases.C3S / 60.0)  # Faster with more alite
    alpha = alpha_ult * (1 - np.exp(-k * age_days ** 0.6))

    # Convert Bogue's wt%-of-clinker phases into kg/m3 masses of that phase.
    clinker_mass_kg_m3 = cement_mass_kg_m3 * clinker_factor
    C3S_kg_m3 = phases.C3S / 100.0 * clinker_mass_kg_m3
    C2S_kg_m3 = phases.C2S / 100.0 * clinker_mass_kg_m3

    # Phase products (simplified stoichiometry, declared placeholders)
    CSH_kg_m3 = alpha * (C3S_kg_m3 + C2S_kg_m3) * 0.7
    CH_kg_m3 = alpha * (C3S_kg_m3 * 0.3 + C2S_kg_m3 * 0.15)

    # Heat of hydration. The phase percentages are wt% OF CLINKER, and the
    # coefficients are J per gram of the pure phase (== kJ/kg), so summing
    # phase_pct * coefficient and dividing by 100 (percent -> fraction) gives
    # kJ per kg of CLINKER. Multiplying by clinker_factor rebases it to kJ per
    # kg of CEMENT, since heat comes only from the clinker fraction of the
    # cement. (Previously `/ 1000.0` -- a factor-of-10 unit bug; R7.5 WP-1
    # finding 1.)
    heat_kJ_per_kg_clinker = alpha * (
        phases.C3S * 500 + phases.C2S * 250 + phases.C3A * 1340 + phases.C4AF * 420
    ) / 100.0
    heat_kJ_per_kg_cement = heat_kJ_per_kg_clinker * clinker_factor

    return HydrationState(
        age_days=age_days,
        degree_of_hydration=alpha,
        CSH_kg_m3=CSH_kg_m3,
        CH_kg_m3=CH_kg_m3,
        heat_kJ_per_kg_cement=heat_kJ_per_kg_cement,
    )

# ============================================================================
# POZZOLANIC / LATENT-HYDRAULIC REACTION MODELS
# ============================================================================
# Two distinct reaction paths (R7.5 WP-1 finding 6), driven by the SCM's
# `reactivity_class` in data/materials.json:
#   - "pozzolanic" (fly ash): CH is a stoichiometric REACTANT, consumed roughly
#     in proportion to the pozzolan mass that has reacted.
#   - "latent-hydraulic" (slag): the material principally self-hydrates: CH
#     ACTIVATES the reaction rather than being consumed mole-for-mole, so its
#     CH draw per kg is far lower than a true pozzolan's.
# Both sets of rate/stoichiometry constants below are DECLARED PLACEHOLDERS --
# there is no calorimetry or degree-of-hydration data in this repo to fit them
# against. Do not tune them to "look right"; that would manufacture false
# precision (see module docstring).
POZZOLANIC_REACTIVITY = {"FLY_ASH_F": 0.3, "FLY_ASH_C": 0.5}
LATENT_HYDRAULIC_REACTIVITY = {"SLAG": 0.8}

# kg CH consumed per kg of reacting material, at full reaction extent (placeholder).
POZZOLANIC_CH_DRAW = 0.2
# An order of magnitude below the pozzolanic draw: slag's CH need is catalytic
# (activation), not stoichiometric consumption of the pozzolan-reaction kind.
LATENT_HYDRAULIC_CH_DRAW = 0.05


def pozzolanic_reaction(
    CH_available_kg_m3: float,
    pozzolan_mass_kg_m3: float,
    pozzolan_type: str,
    age_days: float,
) -> tuple[float, float]:
    """
    Models the pozzolanic reaction: CH + Pozzolan -> C-S-H (true pozzolans, e.g.
    fly ash), where CH is consumed stoichiometrically. All quantities kg/m3.

    Returns:
        (CH_consumed_kg_m3, additional_CSH_kg_m3)
    """
    k = POZZOLANIC_REACTIVITY.get(pozzolan_type, 0.4)

    # Reaction extent (time-dependent)
    extent = min(1.0, k * np.log1p(age_days / 7.0))

    # CH consumed proportional to pozzolan mass and reaction extent, capped by
    # what is actually available (the caller threads a running CH budget across
    # SCMs -- see analyze_mix).
    CH_consumed_kg_m3 = min(CH_available_kg_m3, pozzolan_mass_kg_m3 * POZZOLANIC_CH_DRAW * extent)
    additional_CSH_kg_m3 = CH_consumed_kg_m3 * 1.5

    return CH_consumed_kg_m3, additional_CSH_kg_m3


def latent_hydraulic_reaction(
    CH_available_kg_m3: float,
    material_mass_kg_m3: float,
    material_type: str,
    age_days: float,
) -> tuple[float, float]:
    """
    Models a latent-hydraulic SCM's reaction (e.g. GGBS/slag): the material
    principally self-hydrates, forming C-S-H-like binder in rough proportion to
    its own reacted mass, while CH acts as an activator consumed at a much lower
    rate than a true pozzolan's stoichiometric draw. All quantities kg/m3.

    Returns:
        (CH_consumed_kg_m3, additional_CSH_kg_m3)
    """
    k = LATENT_HYDRAULIC_REACTIVITY.get(material_type, 0.6)

    extent = min(1.0, k * np.log1p(age_days / 7.0))

    CH_consumed_kg_m3 = min(
        CH_available_kg_m3, material_mass_kg_m3 * LATENT_HYDRAULIC_CH_DRAW * extent
    )
    # Self-hydration: CSH formation tracks the material's own reacted mass, not
    # the (much smaller) CH draw -- unlike the pozzolanic path above. Ratio is a
    # declared placeholder.
    additional_CSH_kg_m3 = material_mass_kg_m3 * extent * 0.5

    return CH_consumed_kg_m3, additional_CSH_kg_m3

# ============================================================================
# CARBON FROM CLINKER (More Accurate than Linear Model)
# ============================================================================
# R6.3 — clinker source differentiation. Two clinkers with identical chemistry can
# differ several-fold in carbon depending on HOW they were made: kiln fuel, the
# electricity powering an electrified kiln or a capture plant, and the capture rate.
# The tables below parameterize that; a supplier EPD, when attached, beats all of it
# (measured > modeled — see the resolution order in src/materials.py).
#
# R7.5 — FROZEN SURFACE (through R7.5/WP-1). Everything from here to the end of
# this section (CALCINATION_EF, FUEL_EF, KILN_KWH_PER_KG, GRID_EF,
# DEFAULT_CAPTURE_KWH_PER_T_CO2, clinker_scope_split, carbon_from_clinker,
# embodied_carbon_advanced) is measured and tested (tests/test_chemistry.py,
# tests/test_clinker_source.py). Contrast with the hydration chain above, which
# is uncalibrated by necessity -- this carbon section is not.
#
# R8.0 / WP-B — deliberately revised (see module docstring). `clinker_scope_split`
# and DEFAULT_CAPTURE_KWH_PER_T_CO2 / CALCINATION_EF / FUEL_EF / KILN_KWH_PER_KG /
# GRID_EF stay byte-identical (contract 2 of the spec -- the per-kg-clinker
# worked-example pins in test_clinker_source.py must survive untouched). WP-B
# adds two new per-kg-CEMENT terms inside `carbon_from_clinker` only: process
# electricity (B1) and LC3 constituent carbon (B2). Every resulting change to a
# quoted number is re-derived in the commit message and re-pinned in the owned
# test files, per the spec's "number-change protocol".

CALCINATION_EF = 0.53   # kg CO2 / kg clinker — process chemistry, fuel-independent

# Kiln combustion emission factor (kg CO2 / kg clinker) by fuel. "electric" moves the
# heat demand to electricity (Scope 2) — see KILN_KWH_PER_KG. Waste/biomass factors
# follow common accounting conventions; jurisdictions differ (an EPD's reference
# field should say which convention it uses).
FUEL_EF = {
    "coal": 0.40, "petcoke": 0.42, "natural_gas": 0.28,
    "alt_waste": 0.15, "biomass": 0.05, "electric": 0.0,
}
# Electricity demand (kWh / kg clinker) that the fuel choice moves onto the grid:
# ~3.4 GJ/t clinker of kiln heat ≈ 0.95 kWh/kg for a fully electrified kiln.
KILN_KWH_PER_KG = {"electric": 0.95}

# Electricity emission factors (kg CO2 / kWh) — indicative regional values.
GRID_EF = {
    "grid_EU": 0.25, "grid_US": 0.35, "grid_CN": 0.55,
    "hydro": 0.004, "nuclear": 0.012, "ppa_renewable": 0.02,
}

# Capture electricity demand (kWh per tonne CO2 captured): compression + auxiliaries
# for a heat-integrated capture plant. Site-specific — override in the descriptor.
DEFAULT_CAPTURE_KWH_PER_T_CO2 = 150.0

# R8.0 / WP-B B1 — process electricity for raw-meal prep, kiln drives, and cement
# grinding, per kg CEMENT (not clinker): ~100-120 kWh/t cement is typical for a
# modern dry-process plant (IEA/WBCSD GNR benchmark range); take 110 kWh/t =
# 0.11 kWh/kg as the declared default. This was previously dropped scope in the
# Tier-2 cement term (see the module docstring) -- it belongs to EVERY cement,
# clinker-only or blended, which is why it is charged per kg cement rather than
# folded into `clinker_scope_split` (that function is per kg clinker and frozen).
CEMENT_PROCESS_KWH_PER_KG = 0.11

# R8.0 / WP-B B2 — the ~5% gypsum in LC3-50 (calcined_clay 0.30 + limestone_filler
# 0.15 + clinker 0.50 + gypsum ~0.05, LC3-50 nominal proportions) has no carbon
# record in data/materials.json (gypsum is not a registry material in this repo)
# and is DELIBERATELY EXCLUDED from the constituent carbon term below -- its A1-A3
# factor is small (natural gypsum, low-energy grinding) relative to the terms that
# are counted, but it is an honest omission, not a zero-carbon claim.


def clinker_scope_split(clinker_mass: float, source: dict) -> dict[str, float]:
    """Scope-split clinker carbon (kg CO2) for a clinker source descriptor.

        source = {"kiln_fuel": "natural_gas", "electricity": "hydro",
                  "capture": {"rate": 0.9, "energy_kwh_per_tCO2": 150}}

    scope1 = (calcination + combustion) × (1 − capture_rate)   — the residual stack.
    scope2 = (kiln electricity + capture energy) × grid EF     — the capture energy
             penalty SURVIVES capture: it is charged to the chosen electricity, so
             hydro-powered capture is nearly free while grid-powered capture is not.
    """
    fuel = source.get("kiln_fuel", "coal")
    elec = source.get("electricity", "grid_EU")
    cap = source.get("capture") or {}
    rate = float(cap.get("rate", 0.0))
    cap_kwh_per_t = float(cap.get("energy_kwh_per_tCO2", DEFAULT_CAPTURE_KWH_PER_T_CO2))

    gross_stack = (CALCINATION_EF + FUEL_EF[fuel]) * clinker_mass
    captured = rate * gross_stack
    scope1 = gross_stack - captured
    kiln_kwh = KILN_KWH_PER_KG.get(fuel, 0.0) * clinker_mass
    capture_kwh = (captured / 1000.0) * cap_kwh_per_t
    scope2 = (kiln_kwh + capture_kwh) * GRID_EF[elec]
    return {"scope1": scope1, "scope2": scope2, "total": scope1 + scope2}


def carbon_from_clinker(
    cement_mass: float,
    clinker_factor: float | None = None,
    kiln_fuel_carbon: float = 0.35,
    cement_type: str = "OPC",
    clinker_source: dict | None = None,
) -> float:
    """
    Calculates CO2 emissions from the CEMENT term based on clinker chemistry.

    The main sources are:
    1. Calcination of limestone: CaCO3 → CaO + CO2 (~0.53 kg CO2/kg clinker)
    2. Fuel combustion in the kiln (~0.35 kg CO2/kg clinker, varies by fuel)
    3. (R8.0/WP-B B1) Process electricity for raw-meal prep, kiln drives, and
       grinding, per kg CEMENT -- see CEMENT_PROCESS_KWH_PER_KG.
    4. (R8.0/WP-B B2) Non-clinker constituent carbon (e.g. LC3's calcined clay
       and limestone filler), per kg CEMENT -- see the cement_type's
       `constituents` entry in data/oxide_compositions.json.

    With a `clinker_source` descriptor (R6.3) the fuel/electricity/capture terms are
    computed per source via `clinker_scope_split`; without one, the legacy clinker
    subtotal (0.53 + kiln_fuel_carbon per kg clinker) is byte-for-byte unchanged --
    both paths then add the same B1/B2 per-kg-cement terms on top (R8.0/WP-B: the
    Tier-2 cement term is no longer clinker-only, closing scope that was previously
    dropped -- see the module docstring).

    The process-electricity grid: with a clinker_source descriptor, its
    `"electricity"` key selects the GRID_EF entry (same grid the scope-split uses,
    so a single source descriptor is internally consistent); without one, the
    legacy/default grid is GRID_EF["grid_EU"] (documented default -- no descriptor
    means no better information about where the plant sits).

    Note: this is the cement-only contribution. For a full mix carbon figure on the
    same system boundary as the Tier-1 model, use `embodied_carbon_advanced`.

    Args:
        cement_mass: Mass of cement (kg/m³)
        clinker_factor: Fraction of cement that is clinker. If None, it is read from
            data/oxide_compositions.json for `cement_type` (e.g., 0.95 OPC, 0.50 LC3).
        kiln_fuel_carbon: Carbon intensity of kiln fuel (legacy path only)
        cement_type: Key into the oxide JSON used when clinker_factor is None, AND
            (R8.0/WP-B B2) to look up the `constituents` recipe for non-clinker
            cement components (e.g. LC3's calcined clay + limestone filler). Absent
            for OPC, so B2 contributes 0 for OPC.
        clinker_source: Optional source descriptor (kiln_fuel / electricity / capture).
    """
    if clinker_factor is None:
        clinker_factor = clinker_factor_for(cement_type)
    clinker_mass = cement_mass * clinker_factor
    if clinker_source is not None:
        clinker_co2 = clinker_scope_split(clinker_mass, clinker_source)["total"]
        grid_ef = GRID_EF[clinker_source.get("electricity", "grid_EU")]
    else:
        calcination_co2 = clinker_mass * CALCINATION_EF
        fuel_co2 = clinker_mass * kiln_fuel_carbon
        clinker_co2 = calcination_co2 + fuel_co2
        grid_ef = GRID_EF["grid_EU"]

    # B1 — process electricity, per kg cement, on the same grid as the clinker term.
    process_co2 = cement_mass * CEMENT_PROCESS_KWH_PER_KG * grid_ef

    # B2 — non-clinker constituent carbon (LC3-style blends only; OPC has no
    # `constituents` key so this is exactly 0 and OPC is unaffected).
    oxide_compositions = load_oxide_compositions()
    constituents = oxide_compositions.get(cement_type, {}).get("constituents", {})
    constituent_co2 = sum(
        cement_mass * frac * get_material(material)["carbon"][0]["value"]
        for material, frac in constituents.items()
    )

    return clinker_co2 + process_co2 + constituent_co2


def embodied_carbon_advanced(
    mix: dict[str, float],
    transport_km: float = 0.0,
    cement_type: str = "OPC",
    factors: dict[str, float] | None = None,
    clinker_source: dict | None = None,
    exotic: dict[str, float] | None = None,
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

    `exotic`, if given, is included in the transport mass (see
    `chemistry_simple.transport_carbon`); default `exotic=None` is bit-identical to
    before R7.5 WP-5.
    """
    factors = factors or SIMPLE_CARBON_FACTORS
    carbon = carbon_from_clinker(mix.get("cement", 0.0), cement_type=cement_type,
                                 clinker_source=clinker_source)

    # Non-cement constituents, using the same (editable) Tier-1 factors.
    for component, factor in factors.items():
        if component == "cement":
            continue
        carbon += mix.get(component, 0.0) * factor

    # Same transport heuristic as Tier-1 -- one shared definition (R7.5 WP-5).
    carbon += transport_carbon(mix, transport_km, factors, exotic=exotic)

    return carbon

# ============================================================================
# INVERSE PLANNER (Generative Mode)
# ============================================================================
def inverse_plan_mix(
    target_strength_mpa: float,
    target_carbon_kg: float | None = None,
    max_cost: float | None = None
) -> dict[str, float]:
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
# SCM mix keys -> the material registry key that carries their chemistry hooks
# (data/materials.json `chemistry.oxides_key` / `chemistry.reactivity_class`).
_SCM_MIX_KEYS = ("slag", "ash")


def _scm_chemistry(name: str) -> tuple[str | None, str | None]:
    """(oxides_key, reactivity_class) for an SCM mix key, or (None, None) if the
    registry has no chemistry hooks for it (keeps analyze_mix from crashing on a
    custom/partial materials registry -- WP-1 only reads this data, never writes it)."""
    try:
        chem = get_material(name).get("chemistry", {})
    except (KeyError, FileNotFoundError, OSError, ValueError):
        return None, None
    return chem.get("oxides_key"), chem.get("reactivity_class")


def analyze_mix(
    mix: dict[str, float],
    cement_type: str = "OPC",
    clinker_source: dict | None = None,
    oxide_compositions: dict | None = None,
) -> dict:
    """
    Full molecular-level analysis of a concrete mix.

    UNCALIBRATED (see module docstring): this is dimensionally checked, unit-
    consistent (kg/m3 throughout the hydration chain), and conserves the
    portlandite budget across SCMs -- but it is not fit to any calorimetry or
    degree-of-hydration measurement. Contrast with the carbon figures below
    (`carbon_kg_m3`, via `carbon_from_clinker`), which ARE measured and tested.

    Returns a comprehensive report including:
    - Clinker phases (Bogue), or None with `phases_unavailable_reason` set if
      `cement_type` is unknown or its record is flagged `bogue_valid: false`
      (blended cements -- Bogue does not apply; see `bogue_calculation`).
    - Hydration state (None under the same condition as phases)
    - Carbon breakdown (always available -- clinker_factor_for degrades
      gracefully to a default, and carbon accounting does not need Bogue phases)
    - Pozzolanic / latent-hydraulic contribution, with the CH (portlandite)
      budget threaded as a running pool across slag then ash IN THAT ORDER, so
      Σ CH consumed ≤ CH produced (never double-spent). The result is therefore
      order-dependent -- a real property of this sequential approximation, not
      hidden behind an arbitrary split.

    Args:
        mix: mix quantities, kg/m3 (plus `age` in days).
        cement_type: key into `oxide_compositions` (e.g. "OPC", "LC3").
        clinker_source: optional clinker-source descriptor, forwarded to
            `carbon_from_clinker` (kiln fuel / electricity / capture).
        oxide_compositions: defaults to `load_oxide_compositions()`.
    """
    oxide_compositions = oxide_compositions or load_oxide_compositions()
    cement_mass = mix.get("cement", 300)

    cement_record = oxide_compositions.get(cement_type)
    phases = None
    hydration = None
    phases_unavailable_reason = None
    if cement_record is None:
        phases_unavailable_reason = f"unknown cement_type '{cement_type}'"
    elif cement_record.get("bogue_valid", True) is False:
        phases_unavailable_reason = (
            f"Bogue calculation does not apply to '{cement_type}' "
            "(bogue_valid=false -- blended cement, not a clinker oxide set)"
        )
    else:
        phases = bogue_calculation(cement_record)
        w_c = mix.get("water", 180) / max(cement_mass, 1)
        clinker_factor = clinker_factor_for(cement_type, oxide_compositions)
        hydration = hydration_kinetics(
            phases, w_c, age_days=mix.get("age", 28),
            cement_mass_kg_m3=cement_mass, clinker_factor=clinker_factor,
        )

    # Pozzolanic / latent-hydraulic reactions, conserving the CH budget: thread a
    # running pool through the sequential reactions (slag, then ash) instead of
    # handing the same CH_kg_m3 to both (R7.5 WP-1 finding 3 -- the previous code
    # overspent the pool: each SCM drew against the FULL portlandite produced).
    # Order is documented here as a real, deliberate limitation of the
    # sequential approximation, not hidden behind an arbitrary split.
    ch_remaining_kg_m3 = hydration.CH_kg_m3 if hydration is not None else 0.0
    pozzolanic_csh_kg_m3 = 0.0
    for name in _SCM_MIX_KEYS:
        scm_mass = mix.get(name, 0)
        if scm_mass <= 0:
            continue
        oxides_key, reactivity_class = _scm_chemistry(name)
        oxides_key = oxides_key or name.upper()
        age_days = mix.get("age", 28)
        if reactivity_class == "latent-hydraulic":
            consumed, csh = latent_hydraulic_reaction(
                ch_remaining_kg_m3, scm_mass, oxides_key, age_days)
        else:
            # Default path for anything not explicitly latent-hydraulic
            # (pozzolanic fly ash, or an SCM the registry doesn't classify).
            consumed, csh = pozzolanic_reaction(
                ch_remaining_kg_m3, scm_mass, oxides_key, age_days)
        ch_remaining_kg_m3 -= consumed
        pozzolanic_csh_kg_m3 += csh

    # Carbon from clinker chemistry -- honours cement_type/clinker_source (R7.5
    # WP-1 finding 4; previously hardcoded to OPC with no source descriptor, so
    # an LC3 mix silently reported OPC carbon). Always computable: it only needs
    # a clinker_factor, which clinker_factor_for degrades gracefully for.
    carbon = carbon_from_clinker(cement_mass, cement_type=cement_type,
                                 clinker_source=clinker_source)

    total_CSH_kg_m3 = (hydration.CSH_kg_m3 if hydration is not None else 0.0) + pozzolanic_csh_kg_m3

    return {
        "clinker_phases": phases,
        "phases_unavailable_reason": phases_unavailable_reason,
        "hydration": hydration,
        "total_CSH_kg_m3": total_CSH_kg_m3,
        "carbon_kg_m3": carbon,
        "pozzolanic_CSH_kg_m3": pozzolanic_csh_kg_m3,
        "CH_remaining_kg_m3": ch_remaining_kg_m3,
    }

if __name__ == "__main__":
    # Example usage
    test_mix = {"cement": 350, "slag": 100, "ash": 0, "water": 160, "coarse_agg": 1000, "fine_agg": 750, "age": 28}
    report = analyze_mix(test_mix)

    print("=== Molecular Analysis Report (DIMENSIONALLY CHECKED, UNCALIBRATED) ===")
    print("No calorimetry or degree-of-hydration data backs this layer -- contrast")
    print("with the carbon figure below, which is measured and tested.")
    if report["clinker_phases"] is not None:
        print(f"Clinker Phases: C3S={report['clinker_phases'].C3S:.1f}%, C2S={report['clinker_phases'].C2S:.1f}%")
        print(f"Degree of Hydration (28d): {report['hydration'].degree_of_hydration:.2%}")
        print(f"Heat released: {report['hydration'].heat_kJ_per_kg_cement:.1f} kJ/kg cement")
    else:
        print(f"Clinker phases unavailable: {report['phases_unavailable_reason']}")
    print(f"Total C-S-H (incl. pozzolanic/latent-hydraulic): {report['total_CSH_kg_m3']:.1f} kg/m3")
    print(f"Clinker-based CO2: {report['carbon_kg_m3']:.1f} kg/m³")
