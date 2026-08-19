"""
physical.py - Physical-validity checks for concrete mixes.

A batched mix must fill ~1 m³: Σ(mass_i / density_i) + entrained air ≈ 1. The box
envelope the generators search does NOT enforce this, so they can propose mixes that
are not batchable. This module provides a volume-balance check + repair (R2.2) and a
workability heuristic (R2.3).

R7.5 WP-2: `mix_volume` optionally accounts for exotic admixture volume too (measured:
150 kg/m³ of calcined clay alone displaces ~0.058 m³, more than the entire
VOLUME_TOLERANCE below). The core-only path (`exotic=None`) is unchanged bit-for-bit;
`volume_error` / `repair_volume` / `enforce_volume` stay core-only by design -- wiring
exotic volume into the batchability check or the repair strategy is a deliberate
non-goal here (see docs/specs/R7.5-chemistry-remediation.md WP-2 Non-goals).
"""
from typing import Dict, Optional

from .materials import densities_view, exotic_densities_view

# Absolute densities (kg/m³) — standard values (ACI 211.1 / material data sheets).
# Since R6.1 a view over the material registry (data/materials.json); values unchanged.
DENSITIES = densities_view()
# R7.5 WP-2: densities for the exotic (delta_estimate) admixtures, kept separate from
# the pinned core-only DENSITIES above.
EXOTIC_DENSITIES = exotic_densities_view()
AIR_FRACTION = 0.02  # entrained/entrapped air, ~2% by volume

# Acceptable |volume − 1 m³|. Calibrated against the 1,030 UCI rows: their 95th
# percentile volume_error is ~0.029 and 100% fall under 0.05, so 0.05 admits every
# real mix with margin for the density assumptions.
VOLUME_TOLERANCE = 0.05


def mix_volume(mix: Dict[str, float], exotic: Optional[Dict[str, float]] = None) -> float:
    """Absolute volume (m³) of a per-m³ batch, including entrained air.

    `exotic`, if given, is a {material: kg/m3} dict of exotic admixture dosing (the
    same shape ui_logic/exotics.py use) and adds Σ mass/density for those materials
    on top of the core volume. Default `exotic=None` reproduces the pre-WP-2
    core-only behaviour exactly, so every existing caller (the generators, via
    enforce_volume) is unaffected.
    """
    vol = sum(mix.get(k, 0.0) / rho for k, rho in DENSITIES.items()) + AIR_FRACTION
    if exotic:
        vol += sum(exotic.get(k, 0.0) / rho for k, rho in EXOTIC_DENSITIES.items())
    return vol


def volume_error(mix: Dict[str, float]) -> float:
    """|mix_volume − 1| — how far the mix is from filling exactly one m³."""
    return abs(mix_volume(mix) - 1.0)


def repair_volume(mix: Dict[str, float]) -> Dict[str, float]:
    """
    Scale the aggregates (coarse+fine, preserving their ratio) to close the volume
    balance; fall back to scaling all material masses if aggregates alone cannot.
    Returns a new dict; non-material keys (e.g. age) are preserved untouched.
    """
    m = dict(mix)
    agg_vol = (m.get("coarse_agg", 0.0) / DENSITIES["coarse_agg"]
               + m.get("fine_agg", 0.0) / DENSITIES["fine_agg"])
    others_vol = (mix_volume(m) - AIR_FRACTION) - agg_vol
    target_solids = 1.0 - AIR_FRACTION
    needed_agg_vol = target_solids - others_vol

    if agg_vol > 1e-9 and needed_agg_vol > 0:
        scale = needed_agg_vol / agg_vol
        m["coarse_agg"] = m.get("coarse_agg", 0.0) * scale
        m["fine_agg"] = m.get("fine_agg", 0.0) * scale
        return m

    # Fallback: scale every material mass to hit the target solids+water volume.
    total_solids = mix_volume(m) - AIR_FRACTION
    if total_solids > 1e-9:
        scale = target_solids / total_solids
        for k in DENSITIES:
            if k in m:
                m[k] = m[k] * scale
    return m


def enforce_volume(mix: Dict[str, float], tol: float = VOLUME_TOLERANCE) -> Dict[str, float]:
    """Repair the mix only if it violates the volume balance beyond `tol`; otherwise
    return it unchanged (so already-valid mixes aren't perturbed)."""
    if volume_error(mix) <= tol:
        return dict(mix)
    return repair_volume(mix)


def workability_flag(mix: Dict[str, float]) -> Optional[str]:
    """Heuristic placeability warning, or None. NOT a hard constraint (v1)."""
    binder = mix.get("cement", 0.0) + mix.get("slag", 0.0) + mix.get("ash", 0.0)
    if binder <= 0:
        return None
    wb = mix.get("water", 0.0) / binder
    sp = mix.get("superplasticizer", 0.0)
    if wb < 0.35 and sp < 6.0:
        return "Low water/binder with little superplasticizer — likely unplaceable without more SP."
    if wb > 0.65:
        return "High water/binder — segregation / bleeding risk."
    return None
