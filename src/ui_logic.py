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
)
from .chemistry_advanced import embodied_carbon_advanced
from .exotics import exotic_carbon, exotic_cost, exotic_strength_delta

PARAM_NAMES: List[str] = [
    "cement", "slag", "ash", "water",
    "superplasticizer", "coarse_agg", "fine_agg", "age",
]


def mix_dict(mix) -> Dict[str, float]:
    """Turn an 8-vector into a named mix dict."""
    return {k: float(v) for k, v in zip(PARAM_NAMES, mix)}


def carbon_for_mode(mix: Dict[str, float], advanced: bool) -> float:
    """The single carbon function the whole UI uses; respects the chemistry toggle."""
    return embodied_carbon_advanced(mix) if advanced else calculate_embodied_carbon(mix)


def compute_metrics(
    mix,
    exotic: Dict[str, float],
    costs: Dict[str, float],
    predictor,
    advanced: bool = False,
    exotic_strength: bool = False,
    uncertainty_fn: Optional[Callable] = None,
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
    return {
        "strength": base_strength + delta,
        "exotic_strength": delta,
        "carbon": carbon_for_mode(d, advanced) + exotic_carbon(exotic),
        "cost": calculate_mix_cost(d, costs) + exotic_cost(exotic),
        "curing": estimate_curing_time(d),
        "uncertainty": float(uncertainty_fn(arr)) if uncertainty_fn else None,
    }


def batch_metrics(samples: np.ndarray, costs: Dict[str, float], predictor, advanced: bool = False) -> dict:
    """
    Vectorised strength/carbon/cost for many mixes at once.

    Strength uses a single `predict_batch` call (the previous UI did one predict per
    row -- hundreds of calls per rerun). Carbon/cost are cheap non-model sums.
    """
    samples = np.atleast_2d(np.asarray(samples, dtype=float))
    strengths = predictor.predict_batch(samples)
    carbons = np.array([carbon_for_mode(mix_dict(s), advanced) for s in samples])
    money = np.array([calculate_mix_cost(mix_dict(s), costs) for s in samples])
    return {"strength": strengths, "carbon": carbons, "cost": money}


def scalarized_fitness(
    mix,
    costs: Dict[str, float],
    predictor,
    w_strength: float,
    w_carbon: float,
    w_cost: float,
    advanced: bool = False,
) -> float:
    """Maximise strength, penalise carbon and cost -- the optimizer objective."""
    arr = np.asarray(mix, dtype=float)
    d = mix_dict(arr)
    strength = float(predictor.predict(arr))
    carbon = carbon_for_mode(d, advanced)
    cost = calculate_mix_cost(d, costs)
    return w_strength * strength - w_carbon * carbon - w_cost * cost


def recommend_recipe(
    explorer,
    target_strength: float,
    method: str = "auto",
    carbon_target: Optional[float] = None,
    advanced: bool = False,
    costs: Optional[Dict[str, float]] = None,
) -> dict:
    """
    Return a single recommended mix for a target strength, via the chosen backend.

    - "ga" / "aco": use the metaheuristic designer's single best mix.
    - "auto" / "flow" / "amortized": sample the posterior and pick the draw whose
      predicted strength is closest to the target.

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
            ranked, errors = designer.design(target_strength, carbon_target=carbon_target)
            if errors[0] < best_err:
                best_err, best_arr = float(errors[0]), ranked[0]
        arr = best_arr
    else:
        samples = explorer.sample_posterior(
            target_strength, carbon_target=carbon_target, n_samples=400, method=method
        )
        preds = predictor.predict_batch(samples)
        arr = samples[int(np.argmin(np.abs(preds - target_strength)))]

    d = mix_dict(arr)
    return {
        "mix": arr,
        "params": d,
        "strength": float(predictor.predict(arr)),
        "carbon": carbon_for_mode(d, advanced),
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
