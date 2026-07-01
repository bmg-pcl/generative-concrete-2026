"""
nsga.py - True multi-objective optimization of mix design with NSGA-II / NSGA-III.

Where the amortized flow (src/amortized.py) and the GA/ACO designers answer a
*conditional* question -- "give me mixes that hit strength X" -- NSGA answers the
*multi-objective* question: "what is the whole trade-off surface between strength,
carbon, and cost?" It returns a Pareto front, not a single-target cloud.

The two approaches compose (see docs/WORKFLOW.md):
  * The flow / GA is fast and target-conditioned; NSGA is slower but maps the whole
    front with no target and no scalarization weights.
  * The flow can WARM-START NSGA: seed its initial population with realistic,
    in-envelope mixes near a target so it converges faster and stays in-distribution.

Built on pymoo (https://pymoo.org). Imported lazily-guarded so the rest of the app
works without it.
"""
from typing import Dict, List, Optional

import numpy as np

from .generative_ga import PARAM_NAMES, data_envelope
from .ui_logic import mix_dict, carbon_for_mode
from .chemistry_simple import calculate_mix_cost
from .physical import volume_error, VOLUME_TOLERANCE

try:
    from pymoo.core.problem import Problem
    from pymoo.core.callback import Callback
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.optimize import minimize
    _PYMOO_AVAILABLE = True
except Exception:  # pragma: no cover - only when pymoo is absent
    _PYMOO_AVAILABLE = False


def pymoo_available() -> bool:
    return _PYMOO_AVAILABLE


if _PYMOO_AVAILABLE:

    class MixDesignProblem(Problem):
        """3-objective mix design: MAXIMISE strength, MINIMISE carbon and cost.

        pymoo minimises, so strength enters as -strength. Decision variables are the
        8 mix parameters, box-bounded to the training-data envelope.
        """

        def __init__(self, predictor, bounds, advanced, costs, carbon_kwargs=None, robust=False):
            # Constraints: volume balance (always) + in-support (robust only).
            super().__init__(n_var=len(bounds), n_obj=3, n_ieq_constr=(2 if robust else 1),
                             xl=bounds[:, 0], xu=bounds[:, 1])
            self.predictor = predictor
            self.advanced = advanced
            self.costs = costs
            self.carbon_kwargs = carbon_kwargs or {}
            self.robust = robust
            self.threshold = predictor.support_threshold() if robust else None

        def _evaluate(self, X, out, *args, **kwargs):
            if self.robust:
                # Optimize the conformal lower bound (guaranteed strength).
                strength, _, _ = self.predictor.predict_interval(X)
            else:
                strength = self.predictor.predict_batch(X)
            carbon = np.array([carbon_for_mode(mix_dict(x), self.advanced, **self.carbon_kwargs) for x in X])
            cost = np.array([calculate_mix_cost(mix_dict(x), self.costs) for x in X])
            out["F"] = np.column_stack([-strength, carbon, cost])
            # Physical-validity constraint (<=0 feasible): the front is batchable by
            # construction. In robust mode, also require in-support.
            g_vol = np.array([volume_error(mix_dict(x)) for x in X]) - VOLUME_TOLERANCE
            if self.robust:
                out["G"] = np.column_stack([g_vol, self.predictor.novelty(X) - self.threshold])
            else:
                out["G"] = g_vol

    class _FrontHistory(Callback):
        """Record the best value of each objective per generation for a convergence view."""

        def __init__(self):
            super().__init__()
            self.data["best_strength"] = []
            self.data["min_carbon"] = []
            self.data["min_cost"] = []

        def notify(self, algorithm):
            F = algorithm.pop.get("F")
            self.data["best_strength"].append(float(-F[:, 0].min()))
            self.data["min_carbon"].append(float(F[:, 1].min()))
            self.data["min_cost"].append(float(F[:, 2].min()))


def _seed_sampling(seed_population: Optional[np.ndarray], pop_size: int, bounds: np.ndarray):
    """Build an initial (pop_size, n_var) population from an optional warm-start seed."""
    n_var = len(bounds)
    if seed_population is None or len(seed_population) == 0:
        return None  # let pymoo use its default random sampling
    seed = np.atleast_2d(np.asarray(seed_population, dtype=float))
    if len(seed) >= pop_size:
        return seed[:pop_size]
    # Pad with random in-envelope mixes so the initial population is full.
    n_pad = pop_size - len(seed)
    pad = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_pad, n_var))
    return np.vstack([seed, pad])


def run_nsga(
    predictor,
    advanced: bool = False,
    costs: Optional[Dict[str, float]] = None,
    algorithm: str = "nsga2",
    pop_size: int = 60,
    n_gen: int = 40,
    seed_population: Optional[np.ndarray] = None,
    n_partitions: int = 12,
    random_seed: int = 1,
    param_names: List[str] = PARAM_NAMES,
    bounds: Optional[np.ndarray] = None,
    carbon_kwargs: Optional[dict] = None,
    robust: bool = False,
    age: Optional[float] = None,
) -> Dict:
    """
    Run NSGA-II or NSGA-III and return the Pareto front.

    Args:
        algorithm: "nsga2" or "nsga3".
        seed_population: optional (m, 8) array of warm-start mixes (e.g. from the
            amortized flow at a target strength).
        n_partitions: NSGA-III reference-direction resolution (more = denser front).

    Returns dict with the front mixes and their objective values (natural units),
    plus a per-generation convergence history.
    """
    if not _PYMOO_AVAILABLE:
        raise ImportError("pymoo is not installed. `pip install pymoo` to use NSGA-II/III.")

    bounds = data_envelope(param_names) if bounds is None else np.asarray(bounds, dtype=float)
    if age is not None:  # pin age as a fixed design condition (not a free variable)
        bounds = bounds.copy()
        bounds[param_names.index("age")] = (float(age), float(age))
    problem = MixDesignProblem(predictor, bounds, advanced, costs,
                               carbon_kwargs=carbon_kwargs, robust=robust)
    sampling = _seed_sampling(seed_population, pop_size, bounds)

    if algorithm.lower() == "nsga3":
        ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=n_partitions)
        kwargs = {"ref_dirs": ref_dirs, "pop_size": max(pop_size, len(ref_dirs))}
        if sampling is not None:
            sampling = _seed_sampling(seed_population, kwargs["pop_size"], bounds)
            kwargs["sampling"] = sampling
        algo = NSGA3(**kwargs)
        algo_name = "NSGA-III"
    else:
        kwargs = {"pop_size": pop_size}
        if sampling is not None:
            kwargs["sampling"] = sampling
        algo = NSGA2(**kwargs)
        algo_name = "NSGA-II"

    callback = _FrontHistory()
    res = minimize(problem, algo, ("n_gen", n_gen), seed=random_seed,
                   callback=callback, verbose=False)

    X = np.atleast_2d(res.X)
    F = np.atleast_2d(res.F)
    # Report MEAN strength for display so the numbers match the Compare tab, even when
    # robust mode optimized the lower bound (-F[:,0] is the lower bound in that case).
    strength = predictor.predict_batch(X)
    order = np.argsort(strength)  # sort the front by strength for display
    hist = callback.data
    return {
        "algorithm": algo_name,
        "mixes": X[order],
        "strength": strength[order],
        "carbon": F[order, 1],
        "cost": F[order, 2],
        "history": {
            "best_strength": hist["best_strength"],
            "min_carbon": hist["min_carbon"],
            "min_cost": hist["min_cost"],
        },
        "front_size": len(X),
    }


if __name__ == "__main__":
    from .models import StrengthPredictor

    for algo in ("nsga2", "nsga3"):
        out = run_nsga(StrengthPredictor(), algorithm=algo, pop_size=60, n_gen=25)
        print(f"=== {out['algorithm']} ===  front size {out['front_size']}")
        print(f"  strength {out['strength'].min():.1f}-{out['strength'].max():.1f} MPa | "
              f"carbon {out['carbon'].min():.0f}-{out['carbon'].max():.0f} | "
              f"cost ${out['cost'].min():.0f}-${out['cost'].max():.0f}")
