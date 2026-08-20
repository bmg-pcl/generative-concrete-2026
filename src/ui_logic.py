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
from .chemistry_advanced import embodied_carbon_advanced
from .exotics import exotic_carbon, exotic_cost, exotic_strength_delta
from .physical import workability_flag
from .generative_ga import PARAM_NAMES  # single source of the 8-parameter order


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


def carbon_for_mode(mix: Dict[str, float], advanced: bool, transport_km: float = 0.0,
                    cement_type: str = "OPC", factors: Dict[str, float] = None,
                    clinker_source: Optional[dict] = None,
                    exotic: Optional[Dict[str, float]] = None) -> float:
    """The single carbon function the whole UI uses; respects the chemistry toggle,
    the transport distance, the clinker/cement source (incl. an R6.3 clinker-source
    descriptor — kiln fuel / electricity / capture), and any factor overrides.
    `clinker_source` only affects the advanced tier (the simple tier's cement factor
    already bundles the production route into one number). `exotic`, if given, is
    included in the transport mass (R7.5 WP-5); default `exotic=None` is bit-identical
    to before."""
    if advanced:
        return embodied_carbon_advanced(mix, transport_km=transport_km,
                                        cement_type=cement_type, factors=factors,
                                        clinker_source=clinker_source, exotic=exotic)
    return calculate_embodied_carbon(mix, transport_km=transport_km, factors=factors,
                                     exotic=exotic)


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
    """
    arr = np.asarray(mix, dtype=float)
    d = mix_dict(arr)
    base_strength = float(predictor.predict(arr))
    delta = exotic_strength_delta(exotic, enabled=exotic_strength)
    strength = base_strength + delta
    lo, _, hi = predictor.predict_interval(arr)
    novelty = float(predictor.novelty(arr)[0])
    carbon = carbon_for_mode(d, advanced, exotic=exotic, **(carbon_kwargs or {})) + exotic_carbon(exotic)
    cost = calculate_mix_cost(d, costs) + exotic_cost(exotic)
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
                     exotic: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Per-source carbon contributions (kg CO₂/m³) that sum to the DISPLAYED carbon
    (`compute_metrics(...)["carbon"]`): `carbon_for_mode(...)` plus, when `exotic` is
    given, the exotic admixtures' own carbon factor via the "exotics" line (R8.0
    WP-A A1 -- previously the ticket omitted this term entirely, so a dosed mix's
    ticket TOTAL fell short of the displayed number).

    `exotic`, if given, is ALSO included in the "transport" entry's mass (R7.5 WP-5,
    via the shared `transport_carbon`); default `exotic=None` is bit-identical to
    before.
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
    """
    cfg_carbon = {k: config[k] for k in ("advanced", "transport_km", "cement_type",
                                         "factors", "clinker_source")
                  if k in config}
    costs = config.get("costs") or UNIT_COSTS
    bd = carbon_breakdown(mix, exotic=exotic, **cfg_carbon)

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
    for k, c in costs.items():
        rows.append(f"cost_usd,{k},{mix.get(k, 0.0) * c:.2f}")
    # Provenance: what each factor rests on (epd:REF / database:REF / user-override).
    # A ticket that discloses "placeholder" is honest; one that hides it is a liability.
    for k, src in (config.get("carbon_provenance") or {}).items():
        rows.append(f'provenance,{k},"{src}"')
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
    for k in ("advanced", "cement_type", "transport_km", "robust", "waste_factor"):
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
