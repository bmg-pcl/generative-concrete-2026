"""
ui_logic.py - Pure, testable logic behind the Streamlit UI (app.py).

app.py should only wire widgets to these functions; all number-crunching lives here
so it can be unit-tested without a browser. Keeping it here also guarantees the UI is
*coherent*: there is exactly ONE carbon path, ONE metrics path, and ONE fitness path,
so the "Chemistry Mode" toggle and the exotics switch affect every tab identically.
"""
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from .chemistry_simple import (
    calculate_embodied_carbon,
    calculate_mix_cost,
    estimate_curing_time,
    transport_carbon,
    CARBON_FACTORS,
    UNIT_COSTS,
)
from .chemistry_advanced import embodied_carbon_advanced, analyze_mix
from .exotics import (
    exotic_carbon,
    exotic_cost,
    exotic_strength_delta,
    compliance_warnings,
    EXOTIC_ADMIXTURES,
)
from .materials import (
    carbon_interval,
    factor_uncertainties_view,
    material_transport_carbon,
    load_materials,
)
from .physical import workability_flag
from .thermal import (
    adiabatic_temperature_rise,
    mass_pour_flag,
    curing_days_at_temperature,
    carbonation_co2_bound_kg_m3,
)
from .generative_ga import PARAM_NAMES  # single source of the 8-parameter order

# R8.0 WP-E: the disclosure fields compute_metrics computes once and mix_ticket
# either reads straight off `metrics` (compute_metrics call sites) or -- when
# `metrics` doesn't carry them (e.g. recommend_recipe's own dict, which predates
# this wave) -- rebuilds fresh from `mix`/`config` via `_disclosure_metrics`, the
# same "prefer the caller's metrics, else derive from the ticket's own inputs"
# pattern `carbon_intensity` already established (R8.0 WP-A A3).
DISCLOSURE_KEYS = (
    "carbon_interval_lo", "carbon_interval_hi",
    "delta_t_adiabatic_C", "mass_pour_flag",
    "carbonation_uptake_bound_kg_m3", "curing_maturity_days",
)


def mix_dict(mix) -> Dict[str, float]:
    """Turn an 8-vector into a named mix dict."""
    return {k: float(v) for k, v in zip(PARAM_NAMES, mix)}


def tensile_estimate(fc: float) -> float:
    """Mean axial tensile strength derived from compressive strength (Eurocode 2).

    f_ctm = 0.30·fc^(2/3)              for fc ≤ 50 MPa
          = 2.12·ln(1 + (fc+8)/10)     for fc > 50 MPa

    This is a *correlation* from compressive strength, not an independent prediction;
    label it as derived, and treat it as unvalidated once exotics (esp. fibers) are on.
    """
    fc = max(float(fc), 0.0)
    if fc <= 50.0:
        return 0.30 * fc ** (2.0 / 3.0)
    return 2.12 * np.log(1.0 + (fc + 8.0) / 10.0)


def _merged_carbon_factors(factors: Optional[Dict[str, float]]) -> Dict[str, float]:
    """Core carbon factors merged with every exotic admixture's own carbon factor.

    `materials.carbon_interval`'s `factors` argument resolves BOTH the `mix` term
    AND, when given, the `exotic` dosing dict's materials through the SAME table
    (its docstring convention, R8.0 WP-D3) -- so a caller passing only the core
    7-material `factors` dict would silently price every dosed exotic at 0 in the
    interval. This merge is additive only: an exotic key never overrides an
    explicit core override already present in `factors`."""
    merged = dict(factors) if factors else dict(CARBON_FACTORS)
    for k, props in EXOTIC_ADMIXTURES.items():
        merged.setdefault(k, props["carbon_factor"])
    return merged


def _split_transport(mix: Dict[str, float], transport_km: float,
                     factors: Dict[str, float],
                     exotic: Optional[Dict[str, float]] = None):
    """R8.0 WP-E Decision 2 (per-material transport coexistence): per-material
    registry transport (`materials.material_transport_carbon`) for every material
    that carries a registry `"transport"` block, PLUS the global-km heuristic
    (`chemistry_simple.transport_carbon`) applied ONLY to the mass of materials
    that do NOT carry one (today: water, superplasticizer) -- so no material's
    transport is silently double-counted or dropped. Returns
    `(transport_registry, transport_global)`; the two always sum to the mix's
    total transport carbon under detail mode."""
    registry = load_materials()

    def _has_block(k):
        return bool(registry.get(k, {}).get("transport"))

    factors_without_block = {k: v for k, v in factors.items() if not _has_block(k)}
    exotic_without_block = ({k: v for k, v in exotic.items() if not _has_block(k)}
                            if exotic else None)
    transport_registry = material_transport_carbon(mix, exotic=exotic)
    transport_global = transport_carbon(mix, transport_km, factors_without_block,
                                        exotic=exotic_without_block)
    return transport_registry, transport_global


def _disclosure_metrics(mix: Dict[str, float], exotic: Optional[Dict[str, float]],
                        carbon_kwargs: Optional[dict], carbon_total: float,
                        site_temp_c: float = 20.0) -> dict:
    """R8.0 WP-E: the disclosure-only figures (D3 carbon interval, C1 thermal, C3
    carbonation bound, C2 maturity curing) layered on top of `carbon_total` (the
    figure `compute_metrics`/the ticket actually display). None of these feed back
    into `carbon`/`cost`/`curing` -- they only ever ADD keys/rows.

    Carbon interval: `materials.carbon_interval` gives a total +/- 1.96*sigma band
    from independent per-material relative uncertainty, built on the mass*factor
    sum EXCLUDING transport and (in the advanced tier) the clinker chemistry model
    -- both honestly undmodelled sources of uncertainty per that function's own
    docstring. Rather than expose that partial-sum band (which would NOT bound
    `carbon_total` whenever transport_km > 0 or the advanced tier's clinker term
    diverges from the flat factor), this RE-CENTERS the same sigma on
    `carbon_total`: `carbon_total +/- sigma`, sigma = (raw_hi - raw_lo) / 2. This
    keeps `carbon_interval_lo <= carbon_total <= carbon_interval_hi` true by
    construction, in every mode, while the *width* still reflects the registry's
    declared per-material uncertainty -- transport and the clinker model's own
    uncertainty remain unmodeled inputs to that width, same exclusion as upstream,
    just applied as a band around the number a reader actually sees.

    Every thermal/curing/carbonation figure here is UNCALIBRATED (see
    `thermal.py`'s module docstring) and total-safe: `None` (or `0.0` for the
    carbonation bound) when the hydration chain is unavailable for `cement_type`
    (e.g. LC3), never a raised exception.
    """
    ck = carbon_kwargs or {}
    cement_type = ck.get("cement_type", "OPC")
    clinker_source = ck.get("clinker_source")
    merged_factors = _merged_carbon_factors(ck.get("factors"))
    uncertainties = factor_uncertainties_view()
    raw_lo, raw_hi = carbon_interval(mix, merged_factors, uncertainties, exotic=exotic)
    sigma = (raw_hi - raw_lo) / 2.0

    delta_t = adiabatic_temperature_rise(mix, cement_type=cement_type)
    analysis = analyze_mix(mix, cement_type=cement_type, clinker_source=clinker_source)

    return {
        "carbon_interval_lo": carbon_total - sigma,
        "carbon_interval_hi": carbon_total + sigma,
        "delta_t_adiabatic_C": delta_t,
        "mass_pour_flag": mass_pour_flag(delta_t),
        "carbonation_uptake_bound_kg_m3": carbonation_co2_bound_kg_m3(analysis),
        "curing_maturity_days": curing_days_at_temperature(
            mix, temp_c=site_temp_c, cement_type=cement_type),
    }


def carbon_for_mode(mix: Dict[str, float], advanced: bool, transport_km: float = 0.0,
                    cement_type: str = "OPC", factors: Dict[str, float] = None,
                    clinker_source: Optional[dict] = None,
                    exotic: Optional[Dict[str, float]] = None,
                    transport_detail: bool = False) -> float:
    """The single carbon function the whole UI uses; respects the chemistry toggle,
    the transport distance, the clinker/cement source (incl. an R6.3 clinker-source
    descriptor — kiln fuel / electricity / capture), and any factor overrides.
    `clinker_source` only affects the advanced tier (the simple tier's cement factor
    already bundles the production route into one number). `exotic`, if given, is
    included in the transport mass (R7.5 WP-5); default `exotic=None` is bit-identical
    to before.

    `transport_detail` (R8.0 WP-E Decision 2, default False -- bit-identical):
    when True, the single global-km transport term embedded in the tier functions
    below is swapped for the per-material-registry-plus-partial-global-km split
    (`_split_transport`) -- the same swap `carbon_breakdown` applies to its
    "transport" line, so the two stay reconciled."""
    if advanced:
        base = embodied_carbon_advanced(mix, transport_km=transport_km,
                                        cement_type=cement_type, factors=factors,
                                        clinker_source=clinker_source, exotic=exotic)
    else:
        base = calculate_embodied_carbon(mix, transport_km=transport_km, factors=factors,
                                         exotic=exotic)
    if not transport_detail:
        return base
    factors_eff = factors or CARBON_FACTORS
    old_transport = transport_carbon(mix, transport_km, factors_eff, exotic=exotic)
    transport_registry, transport_global = _split_transport(mix, transport_km, factors_eff, exotic)
    return base - old_transport + transport_registry + transport_global


def compute_metrics(
    mix,
    exotic: Dict[str, float],
    costs: Dict[str, float],
    predictor,
    advanced: bool = False,
    exotic_strength: bool = False,
    uncertainty_fn: Optional[Callable] = None,
    carbon_kwargs: Optional[dict] = None,
    waste_factor: float = 0.0,
    transport_detail: bool = False,
    site_temp_c: float = 20.0,
) -> dict:
    """
    All performance metrics for one mix, on the selected chemistry tier.

    `exotic_strength` gates the (unvalidated) exotic strength contribution; when False
    exotics move only cost and carbon. Carbon/cost always include the exotic terms --
    carbon via both the exotic materials' own factor (`exotic_carbon`) AND their
    transport mass (R7.5 WP-5: `carbon_for_mode`'s `exotic=` thread), since dosed
    admixtures are physically hauled to site too.

    `waste_factor` (R8.0 WP-A A2, default 0.0) is batched-vs-placed overbatch --
    3-8% spillage/over-ordering/pump losses are typical. It scales `carbon`/`cost`
    into `carbon_as_placed`/`cost_as_placed` ONLY: the batched `carbon`/`cost`
    figures -- what the optimizers target -- never move, and at `waste_factor=0.0`
    the as-placed figures are bit-identical to the batched ones.

    `carbon_intensity` (A3) is kg CO2 per m3.MPa of the point-strength estimate --
    the number procurement actually compares across bids. It is NOT fed to any
    optimizer objective (that would change Pareto fronts).

    `transport_detail` (R8.0 WP-E Decision 2, default False -- bit-identical) swaps
    `carbon`'s single global-km transport term for the per-material-registry-plus-
    partial-global-km split (see `carbon_for_mode`).

    `site_temp_c` (R8.0 WP-E Decision 1, default 20.0) only feeds the disclosure-
    only `curing_maturity_days` secondary metric below -- it never touches `curing`
    (the primary, unedited heuristic) or any carbon/cost figure.

    R8.0 WP-E adds five DISCLOSURE-ONLY fields that never feed `carbon`/`cost`/
    `curing`: `carbon_interval_lo`/`hi` (D3), `delta_t_adiabatic_C`/`mass_pour_flag`
    (C1), `carbonation_uptake_bound_kg_m3` (C3, informational, OUTSIDE the A1-A4
    boundary -- never subtracted from `carbon`), and `curing_maturity_days` (C2) --
    a SECONDARY estimate alongside, not a replacement for, `curing`: the two
    measure different things (see `thermal.curing_days_at_temperature`'s
    docstring) and disagreeing is information, not a bug. `curing_maturity_days`
    is `None` whenever the hydration chain is unavailable for `cement_type` (e.g.
    LC3) -- see `_disclosure_metrics`.
    """
    arr = np.asarray(mix, dtype=float)
    d = mix_dict(arr)
    base_strength = float(predictor.predict(arr))
    delta = exotic_strength_delta(exotic, enabled=exotic_strength)
    strength = base_strength + delta
    lo, _, hi = predictor.predict_interval(arr)
    novelty = float(predictor.novelty(arr)[0])
    carbon = (carbon_for_mode(d, advanced, exotic=exotic, transport_detail=transport_detail,
                              **(carbon_kwargs or {}))
             + exotic_carbon(exotic))
    cost = calculate_mix_cost(d, costs) + exotic_cost(exotic)
    disclosure = _disclosure_metrics(d, exotic, carbon_kwargs, carbon, site_temp_c=site_temp_c)
    return {
        "strength": strength,
        "exotic_strength": delta,
        "tensile": tensile_estimate(strength),  # EC2 correlation (derived)
        "interval_lo": float(lo[0]) + delta,   # 90% prediction interval, shifted by
        "interval_hi": float(hi[0]) + delta,   # any exotic strength estimate
        "novelty": novelty,
        "in_support": bool(novelty <= predictor.support_threshold()),
        "workability": workability_flag(d),
        "carbon": carbon,
        "carbon_as_placed": carbon * (1.0 + waste_factor),
        "carbon_intensity": carbon / max(strength, 1.0),
        "cost": cost,
        "cost_as_placed": cost * (1.0 + waste_factor),
        "curing": estimate_curing_time(d),
        "uncertainty": float(uncertainty_fn(arr)) if uncertainty_fn else None,
        **disclosure,
    }


def batch_metrics(samples: np.ndarray, costs: Dict[str, float], predictor, advanced: bool = False,
                  carbon_kwargs: Optional[dict] = None) -> dict:
    """
    Vectorised strength/carbon/cost for many mixes at once.

    Strength uses a single `predict_batch` call (the previous UI did one predict per
    row -- hundreds of calls per rerun). Carbon/cost are cheap non-model sums.
    """
    samples = np.atleast_2d(np.asarray(samples, dtype=float))
    strengths = predictor.predict_batch(samples)
    cf = carbon_kwargs or {}
    carbons = np.array([carbon_for_mode(mix_dict(s), advanced, **cf) for s in samples])
    money = np.array([calculate_mix_cost(mix_dict(s), costs) for s in samples])
    return {"strength": strengths, "carbon": carbons, "cost": money,
            "novelty": predictor.novelty(samples)}


def scalarized_fitness(
    mix,
    costs: Dict[str, float],
    predictor,
    w_strength: float,
    w_carbon: float,
    w_cost: float,
    advanced: bool = False,
    carbon_kwargs: Optional[dict] = None,
    robust: bool = False,
) -> float:
    """Maximise strength, penalise carbon and cost -- the optimizer objective.

    With `robust=True`, the strength term is the conformal lower bound (guaranteed
    strength) and an out-of-support penalty discourages extrapolated mixes."""
    arr = np.asarray(mix, dtype=float)
    d = mix_dict(arr)
    if robust:
        lo, _, _ = predictor.predict_interval(arr)
        strength = float(lo[0])
    else:
        strength = float(predictor.predict(arr))
    carbon = carbon_for_mode(d, advanced, **(carbon_kwargs or {}))
    cost = calculate_mix_cost(d, costs)
    fitness = w_strength * strength - w_carbon * carbon - w_cost * cost
    if robust:
        nov = float(predictor.novelty(arr)[0])
        fitness -= 10.0 * max(0.0, nov - predictor.support_threshold())
    return fitness


def recommend_recipe(
    explorer,
    target_strength: float,
    method: str = "auto",
    carbon_target: Optional[float] = None,
    advanced: bool = False,
    costs: Optional[Dict[str, float]] = None,
    carbon_kwargs: Optional[dict] = None,
    robust: bool = False,
    age: Optional[float] = None,
) -> dict:
    """
    Return a single recommended mix for a target strength, via the chosen backend.

    - "ga" / "aco": use the metaheuristic designer's single best mix.
    - "auto" / "flow" / "amortized": sample the posterior and pick the best draw.

    With `robust=True`, the metaheuristics optimize the conformal lower bound and an
    out-of-support penalty; the sampling backends prefer in-support draws and match on
    the lower bound, so the recommended recipe is one whose *guaranteed* strength meets
    the target and that sits inside the trusted data region.

    Returns the mix vector, its named params, and predicted strength/carbon/cost.
    """
    predictor = explorer.predictor
    if method in ("ga", "aco"):
        # Metaheuristics are stochastic and occasionally under-converge; keep the
        # best candidate over a few cheap restarts so the recommended recipe is
        # reliably close to the target.
        designer = explorer.designer if method == "ga" else explorer.aco_designer
        best_arr, best_err = None, np.inf
        for _ in range(3):
            ranked, errors = designer.design(target_strength, carbon_target=carbon_target,
                                             robust=robust, age=age)
            if errors[0] < best_err:
                best_err, best_arr = float(errors[0]), ranked[0]
        arr = best_arr
    else:
        samples = explorer.sample_posterior(
            target_strength, carbon_target=carbon_target, n_samples=400, method=method,
            robust=robust, age=age,
        )
        if robust:
            lo, _, _ = predictor.predict_interval(samples)
            nov = predictor.novelty(samples)
            in_sup = nov <= predictor.support_threshold()
            score = np.abs(lo - target_strength) + np.where(in_sup, 0.0, 1e3)
            arr = samples[int(np.argmin(score))]
        else:
            preds = predictor.predict_batch(samples)
            arr = samples[int(np.argmin(np.abs(preds - target_strength)))]

    d = mix_dict(arr)
    lo, _, hi = predictor.predict_interval(arr)
    novelty = float(predictor.novelty(arr)[0])
    return {
        "mix": arr,
        "params": d,
        "strength": float(predictor.predict(arr)),
        "interval_lo": float(lo[0]),
        "interval_hi": float(hi[0]),
        "novelty": novelty,
        "in_support": bool(novelty <= predictor.support_threshold()),
        "workability": workability_flag(d),
        "tensile": tensile_estimate(float(predictor.predict(arr))),
        "curing": estimate_curing_time(d),
        "carbon": carbon_for_mode(d, advanced, **(carbon_kwargs or {})),
        "cost": calculate_mix_cost(d, costs) if costs else calculate_mix_cost(d),
    }


def pareto_front_mask(strength, carbon, cost) -> np.ndarray:
    """
    Boolean mask of the non-dominated (Pareto-optimal) points.

    Objectives: MAXIMIZE strength, MINIMIZE carbon, MINIMIZE cost. Point i is
    dominated if some other point is at least as good on all three objectives and
    strictly better on at least one; non-dominated points form the Pareto front.

    O(n^2) but vectorised per point; callers should cap n (a few thousand) since the
    scalarized search can evaluate many points.
    """
    strength = np.asarray(strength, dtype=float)
    carbon = np.asarray(carbon, dtype=float)
    cost = np.asarray(cost, dtype=float)
    n = len(strength)
    on_front = np.ones(n, dtype=bool)
    for i in range(n):
        dominated_by = (
            (strength >= strength[i]) & (carbon <= carbon[i]) & (cost <= cost[i])
            & ((strength > strength[i]) | (carbon < carbon[i]) | (cost < cost[i]))
        )
        if dominated_by.any():
            on_front[i] = False
    return on_front


def carbon_breakdown(mix: Dict[str, float], advanced: bool = False, transport_km: float = 0.0,
                     cement_type: str = "OPC", factors: Dict[str, float] = None,
                     clinker_source: Optional[dict] = None,
                     exotic: Optional[Dict[str, float]] = None,
                     transport_detail: bool = False) -> Dict[str, float]:
    """Per-source carbon contributions (kg CO₂/m³) that sum to the DISPLAYED carbon
    (`compute_metrics(...)["carbon"]`): `carbon_for_mode(...)` plus, when `exotic` is
    given, the exotic admixtures' own carbon factor via the "exotics" line (R8.0
    WP-A A1 -- previously the ticket omitted this term entirely, so a dosed mix's
    ticket TOTAL fell short of the displayed number).

    `exotic`, if given, is ALSO included in the transport entry's mass (R7.5 WP-5,
    via the shared `transport_carbon`); default `exotic=None` is bit-identical to
    before.

    `transport_detail` (R8.0 WP-E Decision 2, default False -- bit-identical): the
    single "transport" line splits into "transport_registry" (per-material,
    registry-block distances) and "transport_global" (the global-km heuristic,
    applied only to materials WITHOUT a registry block) -- the same split
    `carbon_for_mode` applies to its own total, so the two stay reconciled.
    """
    from .chemistry_advanced import carbon_from_clinker
    factors = factors or CARBON_FACTORS
    bd = {}
    for k, f in factors.items():
        if k == "cement" and advanced:
            bd[k] = carbon_from_clinker(mix.get("cement", 0.0), cement_type=cement_type,
                                        clinker_source=clinker_source)
        else:
            bd[k] = mix.get(k, 0.0) * f
    if transport_detail:
        transport_registry, transport_global = _split_transport(mix, transport_km, factors, exotic)
        bd["transport_registry"] = transport_registry
        bd["transport_global"] = transport_global
    else:
        bd["transport"] = transport_carbon(mix, transport_km, factors, exotic=exotic)
    bd["exotics"] = exotic_carbon(exotic) if exotic else 0.0
    return bd


def mix_ticket(mix: Dict[str, float], metrics: dict, config: dict,
              exotic: Optional[Dict[str, float]] = None) -> str:
    """A CSV 'mix ticket' — the recipe + predictions (with interval), carbon and cost
    breakdowns, per-material carbon provenance (which EPD/database/override produced
    each factor), the active config, and the standing disclaimer. The carbon
    breakdown sums exactly to the displayed carbon.

    `exotic` (R8.0 WP-A A1), if given, must be the SAME dosing dict `compute_metrics`
    was called with -- it is threaded into `carbon_breakdown` so the ticket's
    `carbon_kgCO2,TOTAL` row reconciles with the displayed carbon for a dosed mix
    too; default `exotic=None` is bit-identical to before.

    R8.0 WP-E disclosure additions (all unconditional -- they disclose, they never
    change TOTAL): a carbon interval band (D3), per-material allocation basis +
    vintage (D1), compliance warnings for dosed restricted materials (D2), thermal
    ΔT/mass-pour advisory and an informational carbonation-uptake bound (C1/C3),
    a secondary UNCALIBRATED maturity-based curing estimate alongside (not instead
    of) the primary heuristic (C2/Decision 1), and -- when `config["transport_detail"]`
    is on (Decision 2) -- per-material transport mode/km disclosure.
    """
    cfg_carbon = {k: config[k] for k in ("advanced", "transport_km", "cement_type",
                                         "factors", "clinker_source")
                  if k in config}
    transport_detail = bool(config.get("transport_detail", False))
    costs = config.get("costs") or UNIT_COSTS
    bd = carbon_breakdown(mix, exotic=exotic, transport_detail=transport_detail, **cfg_carbon)

    rows = ["section,key,value", f"meta,generated,{config.get('timestamp', '')}"]
    for p in PARAM_NAMES:
        rows.append(f"mix,{p},{mix.get(p, 0.0):.1f}")
    rows += [
        f"prediction,strength_MPa,{metrics['strength']:.1f}",
        f"prediction,interval90_lo,{metrics['interval_lo']:.1f}",
        f"prediction,interval90_hi,{metrics['interval_hi']:.1f}",
        f"prediction,tensile_EC2_MPa,{metrics.get('tensile', 0.0):.2f}",
        f"prediction,curing_days_heuristic,{metrics['curing']:.0f}",
        f"prediction,novelty,{metrics['novelty']:.2f}",
        f"prediction,in_support,{metrics['in_support']}",
    ]
    # A3: carbon-intensity KPI (kg CO2 per m3.MPa, point-strength based) -- the row
    # name discloses that basis. Prefer the caller's own metrics dict (compute_metrics
    # always supplies it); fall back to deriving it from this ticket's own totals for
    # callers (e.g. recommend_recipe's design tickets) that predate this field.
    strength = float(metrics.get("strength", 0.0))
    carbon_intensity = metrics.get("carbon_intensity", sum(bd.values()) / max(strength, 1.0))
    rows.append(f"prediction,carbon_intensity_kg_per_m3MPa_point,{carbon_intensity:.3f}")
    for k, v in bd.items():
        rows.append(f"carbon_kgCO2,{k},{v:.2f}")
    total = sum(bd.values())
    rows.append(f"carbon_kgCO2,TOTAL,{total:.2f}")
    # A2: waste factor (batched -> placed), applied at this layer only -- the
    # batched TOTAL above never moves. Default 0.0 makes TOTAL_as_placed == TOTAL.
    waste_factor = float(config.get("waste_factor", 0.0))
    rows.append(f"carbon_kgCO2,TOTAL_as_placed,{total * (1.0 + waste_factor):.2f}")

    # R8.0 WP-E: disclosure-only figures. Prefer compute_metrics's own values (it
    # already computed these against the SAME `carbon` this ticket's TOTAL
    # reconciles with); recompute fresh from mix/config for callers whose metrics
    # dict predates this wave (e.g. recommend_recipe's design tickets) -- same
    # fallback shape as the carbon_intensity KPI above.
    site_temp_c = float(config.get("site_temp_c", 20.0))
    if "carbon_interval_lo" in metrics:
        disclosure = {k: metrics.get(k) for k in DISCLOSURE_KEYS}
    else:
        disclosure = _disclosure_metrics(mix, exotic, cfg_carbon, total, site_temp_c=site_temp_c)
    # D3: carbon interval -- a total +/- uncertainty band around the SAME TOTAL
    # above (see _disclosure_metrics for why it is centered there), so lo <= TOTAL
    # <= hi always holds.
    rows.append(f"carbon_kgCO2,interval_lo,{disclosure['carbon_interval_lo']:.2f}")
    rows.append(f"carbon_kgCO2,interval_hi,{disclosure['carbon_interval_hi']:.2f}")
    # C3: carbonation-uptake upper bound -- INFORMATIONAL, OUTSIDE the A1-A4 system
    # boundary this ticket's carbon figures otherwise use, and NEVER subtracted
    # from TOTAL above (see thermal.carbonation_co2_bound_kg_m3's docstring).
    rows.append(f"carbon_kgCO2,carbonation_uptake_bound_informational,"
               f"{disclosure['carbonation_uptake_bound_kg_m3']:.2f}")
    rows.append('note,carbonation_uptake_bound,"Upper bound on portlandite '
                're-carbonation only (EN 16757 Annex BB framing) -- OUTSIDE the '
                'A1-A4 product-stage boundary and NOT included in TOTAL above."')
    # C1: thermal mass-pour advisory (both UNCALIBRATED planning signals -- see
    # thermal.py's module docstring). Omitted, not zeroed, when the hydration
    # chain is unavailable for this cement_type (e.g. LC3).
    if disclosure["delta_t_adiabatic_C"] is not None:
        rows.append(f"thermal,delta_t_adiabatic_C,{disclosure['delta_t_adiabatic_C']:.1f}")
    if disclosure["mass_pour_flag"]:
        rows.append(f'thermal,mass_pour_flag,"{disclosure["mass_pour_flag"]}"')
    # C2/Decision 1: SECONDARY maturity-based curing estimate, alongside (never
    # replacing) prediction,curing_days_heuristic above -- the two estimate
    # different quantities and the maturity path inherits an uncalibrated alpha(t);
    # `None` on LC3 (hydration chain unavailable), never silently substituted.
    maturity = disclosure["curing_maturity_days"]
    maturity_str = f"{maturity:.1f}" if maturity is not None else ""
    rows.append(f"prediction,curing_maturity_days_uncalibrated,{maturity_str}")

    for k, c in costs.items():
        rows.append(f"cost_usd,{k},{mix.get(k, 0.0) * c:.2f}")
    # Provenance: what each factor rests on (epd:REF / database:REF / user-override).
    # A ticket that discloses "placeholder" is honest; one that hides it is a liability.
    for k, src in (config.get("carbon_provenance") or {}).items():
        rows.append(f'provenance,{k},"{src}"')
    # D1: allocation basis + vintage per material actually priced above (core
    # constituents from the breakdown, plus any dosed exotic) -- an EN 15804
    # tender challenge turns on exactly which allocation basis backs a factor.
    _NON_MATERIAL_BD_KEYS = {"transport", "transport_registry", "transport_global", "exotics"}
    provenance_materials = {k for k in bd if k not in _NON_MATERIAL_BD_KEYS}
    if exotic:
        provenance_materials |= {k for k, v in exotic.items() if v}
    registry = load_materials()
    for k in sorted(provenance_materials):
        rec = registry.get(k)
        c0 = (rec or {}).get("carbon", [{}])[0]
        if not c0:
            continue
        rows.append(f"allocation,{k},{c0.get('allocation', '')}")
        rows.append(f"vintage,{k},{c0.get('vintage', '')}")
    # D2: compliance advisories for any dosed material carrying a registry
    # restriction (e.g. calcium_chloride in reinforced concrete).
    if exotic:
        for k, amount in exotic.items():
            if not amount:
                continue
            for w in compliance_warnings({k: amount}):
                rows.append(f'warning,{k},"{w}"')
    # Decision 2: per-material transport mode/km, disclosed only when the
    # transport-detail toggle is on (the toggle that also produced the
    # transport_registry/transport_global split in the breakdown above).
    if transport_detail:
        quantities = dict(mix)
        if exotic:
            for k, v in exotic.items():
                quantities[k] = quantities.get(k, 0) + v
        for k, qty in quantities.items():
            if not qty:
                continue
            block = registry.get(k, {}).get("transport")
            if block:
                rows.append(f'transport_detail,{k},"{block["mode"]} {block["km"]}km"')
    # Scope split for a differentiated clinker source (advanced tier, R6.3).
    if config.get("advanced") and config.get("clinker_source"):
        from .chemistry_advanced import clinker_scope_split, clinker_factor_for
        cs = config["clinker_source"]
        clinker_mass = mix.get("cement", 0.0) * clinker_factor_for(config.get("cement_type", "OPC"))
        split = clinker_scope_split(clinker_mass, cs)
        rows += [
            f"clinker_scope,scope1_kgCO2,{split['scope1']:.2f}",
            f"clinker_scope,scope2_kgCO2,{split['scope2']:.2f}",
            f"clinker_scope,kiln_fuel,{cs.get('kiln_fuel', '')}",
            f"clinker_scope,electricity,{cs.get('electricity', '')}",
            f"clinker_scope,capture_rate,{(cs.get('capture') or {}).get('rate', 0.0)}",
        ]
    for k in ("advanced", "cement_type", "transport_km", "robust", "waste_factor",
             "transport_detail", "site_temp_c"):
        if k in config:
            rows.append(f"config,{k},{config[k]}")
    rows.append('disclaimer,,"Design exploration only — validate physically (ASTM/EN) '
                'before any structural use."')
    return "\n".join(rows)


def validate_lab_csv(df) -> Optional[str]:
    """
    Return None if a calibration upload is usable, else a human-readable error.

    Required: the 9 model columns present and numeric. Kept here (not in the UI) so it
    can be tested and reused.
    """
    required = PARAM_NAMES + ["strength"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return f"Missing required column(s): {', '.join(missing)}."
    if len(df) == 0:
        return "The uploaded file has no rows."
    # A column is bad if any value is missing or non-numeric after coercion. This
    # also catches an all-NaN column (which an earlier dropna-then-check-dtype test
    # let through, poisoning the retrain with NaN targets).
    bad = [c for c in required if pd.to_numeric(df[c], errors="coerce").isna().any()]
    if bad:
        return f"Missing or non-numeric values in column(s): {', '.join(bad)}."
    return None


def validate_session_state(data) -> Optional[str]:
    """Return None if an imported session JSON is usable, else an error message."""
    if not isinstance(data, dict):
        return "Session file is not a JSON object."
    for key in ("mix_a", "mix_b", "costs"):
        if key not in data:
            return f"Session file is missing '{key}'."
    for key in ("mix_a", "mix_b"):
        if len(data[key]) != len(PARAM_NAMES):
            return f"'{key}' must have {len(PARAM_NAMES)} values."
    return None
