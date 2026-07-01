"""
generative_ga.py - A simple, transparent generative model for inverse mix design.

This is the honest first-step generator described in docs/FIX_PLAN.md (Phase 3).
It replaces the old noise sampler in bayesian.py, which returned the *same* cloud
regardless of the requested strength.

The idea is deliberately easy to follow:

    1. We already have a trained forward model:  strength = predict(mix).
    2. To design a mix for a TARGET strength, we search for mixes whose predicted
       strength is close to the target -- i.e. we minimise |predict(mix) - target|.
    3. We do that search with the existing GeneticOptimizer (a small GA), over
       bounds CLAMPED TO THE TRAINING DATA ENVELOPE so candidates stay realistic
       (in-distribution) rather than extrapolating.
    4. The "generative" output is not a single answer but the top-K spread of the
       final population -- a set of distinct mixes that all hit the target. That
       spread is our transparent stand-in for a posterior.

No TensorFlow, no black boxes: everything here is a GA over a known objective.
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ga import GeneticOptimizer
from .models import StrengthPredictor
from .chemistry_simple import calculate_embodied_carbon
from .data_fetcher import load_data

# Canonical parameter order (matches the UCI columns and StrengthPredictor inputs).
PARAM_NAMES: List[str] = [
    "cement", "slag", "ash", "water",
    "superplasticizer", "coarse_agg", "fine_agg", "age",
]

# Robust-mode out-of-support penalty weight (MPa-scale). See docs/specs/R1.
OOS_PENALTY_WEIGHT = 10.0


def data_envelope(param_names: List[str] = PARAM_NAMES) -> np.ndarray:
    """
    Per-parameter (min, max) bounds taken from the training data.

    Searching inside this envelope keeps generated mixes in-distribution, which is
    exactly what the old inverse planner failed to do (it emitted water below the
    dataset minimum, forcing the forward model to extrapolate).
    """
    df = load_data()
    return np.array([(df[name].min(), df[name].max()) for name in param_names], dtype=float)


class _InverseDesignerBase:
    """
    Shared plumbing for metaheuristic inverse designers.

    A subclass only has to implement `design()` -- run some optimizer and return the
    final population ranked best-first. Everything else (the objective, turning the
    ranked population into a sample cloud, the single-best-mix convenience) lives
    here, so the GA and ACO variants stay tiny and identical apart from the search.
    """

    def __init__(
        self,
        predictor: Optional[StrengthPredictor] = None,
        param_names: List[str] = PARAM_NAMES,
        bounds: Optional[np.ndarray] = None,
        jitter_frac: float = 0.02,
    ):
        self.predictor = predictor or StrengthPredictor()
        self.param_names = param_names
        self.bounds = data_envelope(param_names) if bounds is None else np.asarray(bounds, dtype=float)
        self.n_dims = len(self.bounds)
        self._age_idx = param_names.index("age")
        # Jitter used when expanding the elite set into a sample cloud, as a
        # fraction of each parameter's range.
        self._jitter = jitter_frac * (self.bounds[:, 1] - self.bounds[:, 0])

    # -- objective -----------------------------------------------------------
    def _make_objective(self, target_strength: float, carbon_target: Optional[float],
                        robust: bool = False):
        """
        Returns a scalar error to MINIMISE:

            error = |strength(mix) - target_strength|            (always)
                  + max(0, carbon(mix) - carbon_target)          (only if carbon_target given)
                  + OOS_PENALTY_WEIGHT·max(0, novelty - thresh)  (robust only)

        With `robust=True`, `strength` is the conformal lower bound (the guaranteed
        strength) rather than the mean, and an out-of-support penalty pulls the search
        away from extrapolated regions the prediction can't be trusted in.
        """
        threshold = self.predictor.support_threshold() if robust else None

        def objective(theta: np.ndarray) -> float:
            if robust:
                lo, _, _ = self.predictor.predict_interval(theta)
                strength = float(lo[0])
            else:
                strength = self.predictor.predict(theta)
            error = abs(strength - target_strength)
            if carbon_target is not None:
                mix = dict(zip(self.param_names, theta))
                carbon = calculate_embodied_carbon(mix)
                error += max(0.0, carbon - carbon_target)
            if robust:
                nov = float(self.predictor.novelty(theta)[0])
                error += OOS_PENALTY_WEIGHT * max(0.0, nov - threshold)
            return float(error)

        return objective

    def _effective_bounds(self, age: Optional[float]) -> np.ndarray:
        """Search bounds with the age dimension pinned to `age` (degenerate bound) when
        a fixed design age is requested, so the optimizer treats age as a condition, not
        a free variable it can exploit (e.g. prescribing a 365-day cure)."""
        if age is None:
            return self.bounds
        b = self.bounds.copy()
        b[self._age_idx] = (float(age), float(age))
        return b

    def _rank(self, optimizer, objective) -> Tuple[np.ndarray, np.ndarray]:
        """Collect an optimizer's final population + global best, ranked best-first."""
        best, _ = optimizer.get_best()
        population = np.vstack([best, optimizer.population])
        errors = np.array([objective(ind) for ind in population])
        order = np.argsort(errors)
        return population[order], errors[order]

    # -- to be provided by subclasses ---------------------------------------
    def design(self, target_strength: float, carbon_target: Optional[float] = None,
               **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    # -- generative interface (shared) --------------------------------------
    def sample(
        self,
        target_strength: float,
        n_samples: int = 2000,
        carbon_target: Optional[float] = None,
        robust: bool = False,
        age: Optional[float] = None,
    ) -> np.ndarray:
        """
        Produce an (n_samples, n_dims) cloud of target-conditioned mixes.

        We take the elite quarter of the final population and resample it with a
        small Gaussian jitter (clipped to the data envelope). This turns a handful
        of good solutions into a smooth spread suitable for the dashboard surface,
        while keeping every sample tied to the requested target.
        """
        ranked, _ = self.design(target_strength, carbon_target, robust=robust, age=age)
        n_elite = max(10, len(ranked) // 4)
        elite = ranked[:n_elite]

        eb = self._effective_bounds(age)
        idx = np.random.randint(0, len(elite), n_samples)
        jitter = np.random.normal(0.0, 1.0, (n_samples, self.n_dims)) * self._jitter
        samples = elite[idx] + jitter
        samples = np.clip(samples, eb[:, 0], eb[:, 1])
        if age is not None:
            samples[:, self._age_idx] = float(age)  # jitter must not un-pin age
        return samples

    def best_mix(self, target_strength: float, carbon_target: Optional[float] = None,
                 robust: bool = False, age: Optional[float] = None) -> Dict[str, float]:
        """Single best mix as a dict -- for one-shot callers like inverse_plan_mix."""
        ranked, _ = self.design(target_strength, carbon_target, robust=robust, age=age)
        return dict(zip(self.param_names, ranked[0]))


class PopulationInverseDesigner(_InverseDesignerBase):
    """
    GA-based inverse designer: given a target strength (and optional carbon target),
    generate a spread of realistic mixes that achieve it.
    """

    def design(
        self,
        target_strength: float,
        carbon_target: Optional[float] = None,
        pop_size: int = 80,
        generations: int = 40,
        robust: bool = False,
        age: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run the GA and return the final population sorted best-first."""
        objective = self._make_objective(target_strength, carbon_target, robust=robust)
        optimizer = GeneticOptimizer(
            objective_fn=objective,
            bounds=self._effective_bounds(age).tolist(),
            pop_size=pop_size,
            maximize=False,  # we are minimising the target-match error
        )
        optimizer.run(generations)
        return self._rank(optimizer, objective)


class AntColonyInverseDesigner(_InverseDesignerBase):
    """
    ACO-based inverse designer: identical interface to the GA variant, but uses
    Ant Colony Optimization for continuous domains (ACO_R, see src/aco.py) as the
    search engine. Useful as an independent metaheuristic to compare against the GA.
    """

    def design(
        self,
        target_strength: float,
        carbon_target: Optional[float] = None,
        n_ants: int = 40,
        archive_size: int = 20,
        generations: int = 40,
        robust: bool = False,
        age: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run ACO_R and return the final solution archive sorted best-first."""
        # Imported lazily so importing this module doesn't require the ACO engine.
        from .aco import AntColonyOptimizer

        objective = self._make_objective(target_strength, carbon_target, robust=robust)
        optimizer = AntColonyOptimizer(
            objective_fn=objective,
            bounds=self._effective_bounds(age).tolist(),
            n_ants=n_ants,
            archive_size=archive_size,
            maximize=False,
        )
        optimizer.run(generations)
        return self._rank(optimizer, objective)


if __name__ == "__main__":
    for name, cls in (("GA", PopulationInverseDesigner), ("ACO", AntColonyInverseDesigner)):
        designer = cls()
        print(f"=== {name} inverse designer ===")
        for target in (25.0, 45.0, 65.0):
            mixes, errors = designer.design(target, generations=40)
            best = mixes[0]
            achieved = designer.predictor.predict(best)
            print(
                f"  target={target:>4.0f} MPa -> achieved={achieved:5.1f} MPa "
                f"(err={errors[0]:.2f}) | cement={best[0]:.0f} slag={best[1]:.0f} "
                f"ash={best[2]:.0f} water={best[3]:.0f} age={best[7]:.0f}"
            )
