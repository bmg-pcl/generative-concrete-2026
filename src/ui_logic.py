"""
ui_logic.py - Pure, testable logic behind the Streamlit UI (app.py).

app.py should only wire widgets to these functions; all number-crunching lives here
so it can be unit-tested without a browser. Keeping it here also guarantees the UI is
*coherent*: there is exactly ONE carbon path, ONE metrics path, and ONE fitness path,
so the "Chemistry Mode" toggle and the exotics switch affect every tab identically.
"""
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .chemistry_simple import (
    calculate_embodied_carbon,
    calculate_mix_cost,
    estimate_curing_time,
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
                    cement_type: str = "OPC", factors: Dict[str, float] = None) -> float:
    """The single carbon function the whole UI uses; respects the chemistry toggle,
    the transport distance, the clinker/cement source, and any factor overrides."""
    if advanced:
        return embodied_carbon_advanced(mix, transport_km=transport_km,
                                        cement_type=cement_type, factors=factors)
    return calculate_embodied_carbon(mix, transport_km=transport_km, factors=factors)


def compute_metrics(
    mix,
    exotic: Dict[str, float],
    costs: Dict[str, float],
    predictor,
    advanced: bool = False,
    exotic_strength: bool = False,
    uncertainty_fn: Optional[Callable] = None,
    carbon_kwargs: Optional[dict] = None,
) -> dict:
    """
    All performance metrics for one mix, on the selected chemistry tier.

    `exotic_strength` gates the (unvalidated) exotic strength contribution; when False
    exotics move only cost and carbon. Carbon/cost always include the exotic terms.
    """
    arr = np.asarray(mix, dtype=float)
    d = mix_dict(arr)
    base_strength = float(predictor.predict(arr))
    delta = exotic_strength_delta(exotic, enabled=exotic_strength)
    lo, _, hi = predictor.predict_interval(arr)
    novelty = float(predictor.novelty(arr)[0])
    return {
        "strength": base_strength + delta,
        "exotic_strength": delta,
        "tensile": tensile_estimate(base_strength + delta),  # EC2 correlation (derived)
        "interval_lo": float(lo[0]) + delta,   # 90% prediction interval, shifted by
        "interval_hi": float(hi[0]) + delta,   # any exotic strength estimate
        "novelty": novelty,
        "in_support": bool(novelty <= predictor.support_threshold()),
        "workability": workability_flag(d),
        "carbon": carbon_for_mode(d, advanced, **(carbon_kwargs or {})) + exotic_carbon(exotic),
        "cost": calculate_mix_cost(d, costs) + exotic_cost(exotic),
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
                     cement_type: str = "OPC", factors: Dict[str, float] = None) -> Dict[str, float]:
    """Per-source carbon contributions (kg CO₂/m³) that sum to carbon_for_mode(...)."""
    from .chemistry_advanced import carbon_from_clinker
    factors = factors or CARBON_FACTORS
    bd = {}
    for k, f in factors.items():
        if k == "cement" and advanced:
            bd[k] = carbon_from_clinker(mix.get("cement", 0.0), cement_type=cement_type)
        else:
            bd[k] = mix.get(k, 0.0) * f
    total_mass = sum(mix.get(k, 0.0) for k in factors)
    bd["transport"] = (total_mass / 1000.0) * transport_km * 0.1
    return bd


def mix_ticket(mix: Dict[str, float], metrics: dict, config: dict) -> str:
    """A CSV 'mix ticket' — the recipe + predictions (with interval), carbon and cost
    breakdowns, the active config, and the standing disclaimer. The carbon breakdown
    sums exactly to the displayed carbon."""
    cfg_carbon = {k: config[k] for k in ("advanced", "transport_km", "cement_type", "factors")
                  if k in config}
    costs = config.get("costs") or UNIT_COSTS
    bd = carbon_breakdown(mix, **cfg_carbon)

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
    for k, v in bd.items():
        rows.append(f"carbon_kgCO2,{k},{v:.2f}")
    rows.append(f"carbon_kgCO2,TOTAL,{sum(bd.values()):.2f}")
    for k, c in costs.items():
        rows.append(f"cost_usd,{k},{mix.get(k, 0.0) * c:.2f}")
    for k in ("advanced", "cement_type", "transport_km", "robust"):
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
