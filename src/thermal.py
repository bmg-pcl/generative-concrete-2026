"""
thermal.py - R8.0 WP-C: consumers of the hydration layer.

The Bogue -> hydration -> pozzolanic chain in `chemistry_advanced.py` is
DIMENSIONALLY CHECKED but UNCALIBRATED (see that module's docstring): there is
no calorimetry or degree-of-hydration data anywhere in this repo to fit its
reaction-rate/stoichiometry coefficients against. Every public function in this
module INHERITS that status. Their outputs are order-of-magnitude PLANNING
SIGNALS -- bounds and advisory flags derived from a physically-structured but
unfitted model -- never predictions, and never a substitute for calorimetry,
maturity testing, or a carbonation study on the actual mix.

This module is additive-only (R8.0 spec, WP-C): it imports the hydration chain
as-is and adds no new coefficients to it. It does not wire into the UI or the
mix ticket -- that is Wave B's decision, made with its own displayed-numbers
review (see `chemistry_simple.estimate_curing_time`'s docstring note below on
why that heuristic is not touched here).

Three families of function:

  C1 -- adiabatic_temperature_rise / mass_pour_flag: how hot a mass pour of
        this mix could get if it hydrated with no heat loss, and an advisory
        flag styled after `physical.workability_flag`.
  C2 -- equivalent_age / curing_days_at_temperature: an ASTM C1074 Arrhenius
        maturity function, and its use to rescale the (UNCALIBRATED) hydration
        model's own degree-of-hydration curve for a curing temperature other
        than the model's implicit ~20 C reference.
  C3 -- carbonation_co2_bound_kg_m3: a strict UPPER BOUND (not an estimate) on
        how much CO2 the portlandite left in a mix could re-absorb.

Every function here is TOTAL-SAFE on the "no Bogue/hydration available" case
(blended cements flagged `bogue_valid: false` in data/oxide_compositions.json,
e.g. LC3) -- it degrades to `None` (or `0.0` for the C3 bound, which is
honestly zero when there is no portlandite pool to speak of) with the reason
documented, and never raises.
"""
from __future__ import annotations

import math

from .chemistry_advanced import (
    ClinkerPhases,
    HydrationState,
    bogue_calculation,
    clinker_factor_for,
    hydration_kinetics,
    load_oxide_compositions,
)

# ============================================================================
# Declared constants (R8.0 WP-C)
# ============================================================================
CONCRETE_DENSITY = 2400.0   # kg/m3 -- typical normal-weight concrete bulk density
SPECIFIC_HEAT = 1.0         # kJ/(kg.K) -- typical concrete specific heat (~0.9-1.1 lit. range)

# "Large enough" age (days) that `hydration_kinetics`'s Avrami-style
# `1 - exp(-k * age**0.6)` term has saturated to its asymptote for every phase
# composition this repo ships (k is smallest, ~0.10, for a low-C3S oxide set;
# even then k * 36500**0.6 ~= 43, so exp(-43) is machine-zero). This stands in
# for the mathematical limit age -> infinity, i.e. alpha -> alpha_ult.
_FULL_HYDRATION_AGE_DAYS = 36_500.0

# ASTM C1074 default activation energy: SI unit is J/mol in the Arrhenius term,
# so the kJ/mol argument is converted once here.
_GAS_CONSTANT_J_PER_MOL_K = 8.314

# Molar mass ratio CO2 (44.01 g/mol) / Ca(OH)2 (74.09 g/mol) -- the
# stoichiometry of Ca(OH)2 + CO2 -> CaCO3 + H2O is 1:1 by mole, so this ratio
# converts a kg mass of portlandite into the kg mass of CO2 it could absorb.
_CO2_OVER_CH_MOLAR_RATIO = 44.01 / 74.09


def _hydration_inputs(mix: dict, cement_type: str, oxide_compositions: dict | None = None):
    """Shared setup: (phases, clinker_factor, cement_mass, w_c, reason).

    Mirrors `chemistry_advanced.analyze_mix`'s "is Bogue available for this
    cement_type" guard exactly (unknown type, or a registry record flagged
    `bogue_valid: false` for a blended cement) so every consumer here degrades
    on precisely the same LC3-style inputs that make `analyze_mix` report
    `hydration=None`. On success `phases` is a `ClinkerPhases` and `reason` is
    `None`; on failure `phases` is `None` and `reason` names why.
    """
    oxide_compositions = oxide_compositions or load_oxide_compositions()
    cement_record = oxide_compositions.get(cement_type)
    cement_mass = mix.get("cement", 300)

    if cement_record is None:
        return None, None, cement_mass, None, f"unknown cement_type '{cement_type}'"
    if cement_record.get("bogue_valid", True) is False:
        return None, None, cement_mass, None, (
            f"Bogue calculation does not apply to '{cement_type}' "
            "(bogue_valid=false -- blended cement, not a clinker oxide set)"
        )

    phases = bogue_calculation(cement_record)
    clinker_factor = clinker_factor_for(cement_type, oxide_compositions)
    w_c = mix.get("water", 180) / max(cement_mass, 1)
    return phases, clinker_factor, cement_mass, w_c, None


# ============================================================================
# C1 -- Adiabatic temperature rise + mass-pour flag
# ============================================================================
def adiabatic_temperature_rise(mix: dict, cement_type: str = "OPC") -> float | None:
    """UNCALIBRATED planning signal: adiabatic temperature rise (deg C) if this
    mix's cement hydrated fully with no heat loss to the environment (a mass
    pour approximates this; a thin slab does not -- real pours lose heat and
    will rise less than this bound suggests).

    Derivation: full-hydration heat (`hydration_kinetics` evaluated at an age
    large enough that degree-of-hydration alpha -> alpha_ult, i.e. the
    UNCALIBRATED hydration model's own asymptote for this cement's phase mix)
    times the mix's cement dosage, divided by the thermal mass of one m3 of
    concrete (density x specific heat):

        delta_T = heat_kJ_per_kg_cement * cement_kg_m3 / (CONCRETE_DENSITY * SPECIFIC_HEAT)

    Sanity-check order of magnitude (see tests/test_thermal.py for the
    honestly-measured value, which runs a bit hotter than this back-of-envelope
    sketch because the model's full-hydration heat is nearer 470 than 300
    kJ/kg cement): ~300 kJ/kg x 350 kg/m3 / 2400 ~= 44 C.

    Returns `None` (never raises) when `cement_type` has no Bogue-valid oxide
    record (e.g. LC3, or any registry entry flagged `bogue_valid: false`) --
    the same condition under which `chemistry_advanced.analyze_mix` reports
    `hydration=None`. Returns `0.0` for a zero-cement mix (no heat source).
    """
    phases, clinker_factor, cement_mass, w_c, _reason = _hydration_inputs(mix, cement_type)
    if phases is None:
        return None
    if cement_mass <= 0:
        return 0.0

    state: HydrationState = hydration_kinetics(
        phases, w_c, age_days=_FULL_HYDRATION_AGE_DAYS,
        cement_mass_kg_m3=cement_mass, clinker_factor=clinker_factor,
    )
    return state.heat_kJ_per_kg_cement * cement_mass / (CONCRETE_DENSITY * SPECIFIC_HEAT)


def mass_pour_flag(delta_t: float | None, threshold: float = 20.0) -> str | None:
    """Advisory mass-pour warning, styled after `physical.workability_flag`:
    NOT a hard constraint, a heuristic prompt to consider thermal control
    measures (cooling pipes, low-heat cement, staged lifts) on a large pour.

    ACI 207.2R guidance flags a core-to-surface (or core-to-ambient, in this
    module's adiabatic-bound framing) temperature DIFFERENTIAL of roughly
    19-20 C as the point where thermal-cracking risk becomes a real design
    concern; `threshold` defaults to that ~20 C figure. `delta_t` here is the
    adiabatic (no-loss) bound from `adiabatic_temperature_rise`, an upper
    bound on the real differential a mass pour would see -- so this flag is
    conservative (over-warns relative to a pour that does lose heat), which is
    the right direction for an advisory.

    Returns `None` (never raises) when `delta_t` is `None` (Bogue unavailable
    for this cement_type) -- there is nothing to advise on.
    """
    if delta_t is None:
        return None
    if delta_t > threshold:
        return (
            f"Adiabatic temperature rise ~{delta_t:.0f} C exceeds the ACI 207 "
            f"~{threshold:.0f} C differential guidance -- consider mass-pour "
            "thermal control (low-heat cement, cooling, staged lifts)."
        )
    return None


# ============================================================================
# C2 -- Maturity-based curing estimate (ASTM C1074)
# ============================================================================
def equivalent_age(
    t_days: float,
    temp_c: float,
    ref_c: float = 20.0,
    activation_kj_mol: float = 41.6,
) -> float:
    """ASTM C1074 Arrhenius equivalent age (days): the age, AT THE REFERENCE
    TEMPERATURE `ref_c`, that gives the same cumulative "maturity" as `t_days`
    spent at `temp_c`.

        t_e = t * exp[ -Ea/R * (1/T - 1/T_ref) ]      (T, T_ref in Kelvin)

    `activation_kj_mol` (Ea) defaults to 41.6 kJ/mol, the declared mid-range of
    ASTM C1074's commonly-cited 40-45 kJ/mol band for a general-purpose OPC --
    not fit to any cement in this repo (there is no calorimetry data here to
    fit it against; see the module docstring).

    At `temp_c == ref_c` this returns `t_days` exactly (the exponent is 0).
    Monotonically increasing in `temp_c` (warmer curing matures faster).
    """
    t_kelvin = temp_c + 273.15
    ref_kelvin = ref_c + 273.15
    ea_j_mol = activation_kj_mol * 1000.0
    exponent = -(ea_j_mol / _GAS_CONSTANT_J_PER_MOL_K) * (1.0 / t_kelvin - 1.0 / ref_kelvin)
    return t_days * math.exp(exponent)


def _invert_alpha_to_age(
    phases: ClinkerPhases,
    w_c: float,
    cement_mass: float,
    clinker_factor: float,
    target_alpha: float,
    max_days: float = 1_000_000.0,
) -> float | None:
    """Numerically invert `hydration_kinetics`'s degree_of_hydration(age) for
    the age reaching `target_alpha`, by bisection. Degree of hydration is
    monotonically non-decreasing in age for this model (Avrami-style
    saturation), so bisection is well-posed once an upper bracket is found.

    Deliberately numeric rather than a closed-form inversion of the current
    Avrami shape: `hydration_kinetics` is documented as a placeholder model
    (R7.5), and this inverter should keep working unchanged if its functional
    form is later refit or replaced, per the spec's "keep it generalised" note
    for the annealing/ACO consumers still to come.

    Returns `None` (never raises) if `target_alpha` cannot be reached within
    `max_days` (e.g. `target_alpha` at or beyond `alpha_ult`, which is only
    approached in the mathematical limit, never reached at finite age).
    """
    if target_alpha <= 0:
        return 0.0

    def alpha_at(age_days: float) -> float:
        return hydration_kinetics(
            phases, w_c, age_days=age_days,
            cement_mass_kg_m3=cement_mass, clinker_factor=clinker_factor,
        ).degree_of_hydration

    lo, hi = 1e-6, 1.0
    while alpha_at(hi) < target_alpha:
        if hi >= max_days:
            return None
        hi = min(hi * 2.0, max_days)

    for _ in range(60):  # bisection: 2**-60 relative precision, far more than needed
        mid = 0.5 * (lo + hi)
        if alpha_at(mid) < target_alpha:
            lo = mid
        else:
            hi = mid
    return hi


def curing_days_at_temperature(
    mix: dict,
    target_alpha_fraction: float = 0.7,
    temp_c: float = 20.0,
    cement_type: str = "OPC",
) -> float | None:
    """UNCALIBRATED planning signal: real curing days at a site temperature
    `temp_c` for this mix's (UNCALIBRATED) hydration model to reach
    `target_alpha_fraction * alpha_ult` degree of hydration -- the
    physically-grounded successor to `chemistry_simple.estimate_curing_time`
    (which is NOT edited or rewired by this module; the switchover is a Wave B
    decision made with its own displayed-numbers review, per the R8.0 spec).

    Two steps:
      1. Numerically invert the hydration model's alpha(age) at the model's
         implicit reference temperature (~20 C, ASTM C1074's usual maturity
         reference) for the equivalent age `t_e` reaching the target alpha.
      2. Rescale `t_e` by the ASTM C1074 Arrhenius rate factor for `temp_c`
         relative to that same 20 C reference (`equivalent_age(1.0, temp_c)`),
         since a day at a warmer site temperature matures the mix faster than
         a day at the reference temperature: `t_real = t_e / rate_factor(temp_c)`.

    At `temp_c == 20.0` the rate factor is 1.0, so this equals the direct
    alpha-inversion `t_e` exactly. Decreases monotonically as `temp_c` rises
    (Arrhenius: warmer cures faster).

    Returns `None` (never raises) when Bogue is unavailable for `cement_type`
    (e.g. LC3) or the target cannot be reached in finite time.
    """
    _MATURITY_REF_C = 20.0

    phases, clinker_factor, cement_mass, w_c, _reason = _hydration_inputs(mix, cement_type)
    if phases is None:
        return None

    alpha_ult = min(1.0, w_c / 0.38)
    target_alpha = target_alpha_fraction * alpha_ult

    t_equivalent = _invert_alpha_to_age(phases, w_c, cement_mass, clinker_factor, target_alpha)
    if t_equivalent is None:
        return None

    rate_factor = equivalent_age(1.0, temp_c, ref_c=_MATURITY_REF_C)
    if rate_factor <= 0:
        return None
    return t_equivalent / rate_factor


# ============================================================================
# C3 -- Carbonation-uptake upper bound
# ============================================================================
def carbonation_co2_bound_kg_m3(analysis: dict) -> float:
    """A strict UPPER BOUND (kg CO2/m3), not an estimate, on how much CO2 the
    portlandite (Ca(OH)2) remaining in an `analyze_mix(...)` result could
    re-absorb via carbonation, from the 1:1 molar reaction
    Ca(OH)2 + CO2 -> CaCO3 + H2O:

        bound = analysis["CH_remaining_kg_m3"] * (44.01 / 74.09)

    What this IS NOT, precisely, because an uncalibrated model can bound
    honestly even where it cannot predict:
      - NOT a carbonation rate or a time-to-carbonate -- no diffusion/exposure
        model is attempted here (out of scope, R8.0 spec).
      - NOT inclusive of C-S-H carbonation, which is a real and often larger
        contributor in practice but has no stoichiometric handle in this
        model's simplified C-S-H representation -- excluded, not assumed zero.
      - Reported at the informational level, OUTSIDE the A1-A4 product-stage
        accounting boundary this repo's carbon figures otherwise use, in the
        EN 16757 Annex BB "additional information beyond the system boundary"
        frame -- never add it to `carbon_kg_m3` as an offset.

    Degrades to `0.0` (never raises, and honestly so -- no portlandite pool
    means no bound to report) when `analysis["CH_remaining_kg_m3"]` is `0.0`
    or missing, which is exactly what `analyze_mix` returns both when
    `hydration is None` (e.g. LC3 -- Bogue unavailable, so no CH was ever
    produced to begin with) and when reactive SCMs (slag/ash) have fully
    consumed the portlandite pool.
    """
    ch_remaining = analysis.get("CH_remaining_kg_m3") or 0.0
    return max(0.0, ch_remaining) * _CO2_OVER_CH_MOLAR_RATIO
