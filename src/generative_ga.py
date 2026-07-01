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
        # Jitter used when expanding the elite set into a sample cloud, as a
        # fraction of each parameter's range.
        self._jitter = jitter_frac * (self.bounds[:, 1] - self.bounds[:, 0])

    # -- objective -----------------------------------------------------------
    def _make_objective(self, target_strength: float, carbon_target: Optional[float]):
        """
        Returns a scalar error to MINIMISE:

            error = |predict(mix) - target_strength|            (always)
                  + max(0, carbon(mix) - carbon_target)         (only if carbon_target given)

        The carbon term is a one-sided penalty: it only pushes back when a mix
        exceeds the carbon budget, so it never fights the strength objective for
        already-clean mixes.
        """
        def objective(theta: np.ndarray) -> float:
            strength = self.predictor.predict(theta)
            error = abs(strength - target_strength)
            if carbon_target is not None:
                mix = dict(zip(self.param_names, theta))
                carbon = calculate_embodied_carbon(mix)
                error += max(0.0, carbon - carbon_target)
            return float(error)

        return objective

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
    ) -> np.ndarray:
        """
        Produce an (n_samples, n_dims) cloud of target-conditioned mixes.

        We take the elite quarter of the final population and resample it with a
        small Gaussian jitter (clipped to the data envelope). This turns a handful
        of good solutions into a smooth spread suitable for the dashboard surface,
        while keeping every sample tied to the requested target.
        """
        ranked, _ = self.design(target_strength, carbon_target)
        n_elite = max(10, len(ranked) // 4)
        elite = ranked[:n_elite]

        idx = np.random.randint(0, len(elite), n_samples)
        jitter = np.random.normal(0.0, 1.0, (n_samples, self.n_dims)) * self._jitter
        samples = elite[idx] + jitter
        return np.clip(samples, self.bounds[:, 0], self.bounds[:, 1])

    def best_mix(self, target_strength: float, carbon_target: Optional[float] = None) -> Dict[str, float]:
        """Single best mix as a dict -- for one-shot callers like inverse_plan_mix."""
        ranked, _ = self.design(target_strength, carbon_target)
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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run the GA and return the final population sorted best-first."""
        objective = self._make_objective(target_strength, carbon_target)
        optimizer = GeneticOptimizer(
            objective_fn=objective,
            bounds=self.bounds.tolist(),
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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run ACO_R and return the final solution archive sorted best-first."""
        # Imported lazily so importing this module doesn't require the ACO engine.
        from .aco import AntColonyOptimizer

        objective = self._make_objective(target_strength, carbon_target)
        optimizer = AntColonyOptimizer(
            objective_fn=objective,
            bounds=self.bounds.tolist(),
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
