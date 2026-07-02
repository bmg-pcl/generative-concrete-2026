"""
aco.py - Ant Colony Optimization for continuous domains (ACO_R).

The classic Ant Colony algorithm is defined on discrete graphs. ACO_R (Socha &
Dorigo, 2008) extends it to continuous search spaces, which is what we need for
mix design (8 real-valued parameters).

The idea in one paragraph:
  Instead of a pheromone matrix over edges, ACO_R keeps an *archive* of the k best
  solutions found so far -- that archive IS the pheromone. Each new "ant" builds a
  solution dimension by dimension: it picks one archived solution (better-ranked
  ones are more likely) and samples a Gaussian centred on that solution's value,
  with a spread equal to how far the other archived solutions sit from it. Good
  regions therefore get sampled tightly (exploitation) while still-diverse regions
  get sampled widely (exploration). New solutions are merged into the archive and
  only the best k survive.

It implements the same `Optimizer` interface as the GA, so it drops straight into
the inverse designer as an alternative metaheuristic.

Reference: Socha, K. & Dorigo, M. (2008), "Ant colony optimization for continuous
domains", European Journal of Operational Research 185(3).
"""
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from .base_optimizer import Optimizer


class AntColonyOptimizer(Optimizer):
    """ACO_R: ant colony optimization over a continuous, box-bounded search space."""

    def __init__(
        self,
        objective_fn: Callable[[np.ndarray], float],
        bounds: List[Tuple[float, float]],
        archive_size: int = 20,
        n_ants: int = 40,
        q: float = 0.2,
        xi: float = 0.85,
        maximize: bool = True,
    ):
        super().__init__(objective_fn, bounds, maximize)
        self.k = archive_size          # solution archive size (the "pheromone")
        self.n_ants = n_ants           # new solutions sampled per iteration
        self.q = q                     # locality: small q => favour the very best
        self.xi = xi                   # spread factor (like pheromone evaporation)
        self.dim = len(self.bounds)
        self.generation = 0

        # Initialise the archive with uniform random solutions, then rank it.
        self.archive = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1], (self.k, self.dim)
        )
        self.archive_fit = np.array([self.objective_fn(s) for s in self.archive])
        self._sort_archive()

        self.best_solution = self.archive[0].copy()
        self.best_fitness = float(self.archive_fit[0])

        # Gaussian kernel weights over ranks 1..k (rank 1 = best).
        ranks = np.arange(1, self.k + 1)
        self.weights = np.exp(-((ranks - 1) ** 2) / (2.0 * (self.q * self.k) ** 2))
        self.weights /= self.weights.sum()

    # -- helpers -------------------------------------------------------------
    def _rank_order(self, fitness: np.ndarray) -> np.ndarray:
        order = np.argsort(fitness)
        return order[::-1] if self.maximize else order

    def _sort_archive(self):
        order = self._rank_order(self.archive_fit)
        self.archive = self.archive[order]
        self.archive_fit = self.archive_fit[order]

    def _is_better(self, a: float, b: float) -> bool:
        return a > b if self.maximize else a < b

    # -- core iteration ------------------------------------------------------
    def step(self) -> Dict[str, Any]:
        """Sample n_ants new solutions, merge into the archive, keep the best k."""
        new = np.empty((self.n_ants, self.dim))
        for a in range(self.n_ants):
            # Each ant follows one archived solution (chosen by rank weight).
            guide = np.random.choice(self.k, p=self.weights)
            for i in range(self.dim):
                mu = self.archive[guide, i]
                # Spread = xi * average distance from the guide to the archive in dim i.
                sigma = self.xi * np.abs(self.archive[:, i] - mu).sum() / max(self.k - 1, 1)
                new[a, i] = np.random.normal(mu, sigma if sigma > 0 else 1e-6)
            new[a] = np.clip(new[a], self.bounds[:, 0], self.bounds[:, 1])

        new_fit = np.array([self.objective_fn(s) for s in new])

        # Merge archive + new ants, keep the best k.
        pooled = np.vstack([self.archive, new])
        pooled_fit = np.concatenate([self.archive_fit, new_fit])
        keep = self._rank_order(pooled_fit)[: self.k]
        self.archive, self.archive_fit = pooled[keep], pooled_fit[keep]

        if self._is_better(self.archive_fit[0], self.best_fitness):
            self.best_fitness = float(self.archive_fit[0])
            self.best_solution = self.archive[0].copy()

        self.generation += 1
        stats = {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "avg_fitness": float(np.mean(self.archive_fit)),
            "archive_std": float(np.std(self.archive_fit)),
        }
        self.history.append(stats)
        return stats

    def run(self, iterations: int) -> Tuple[np.ndarray, float]:
        for _ in range(iterations):
            self.step()
        return self.get_best()

    def get_best(self) -> Tuple[np.ndarray, float]:
        return self.best_solution, self.best_fitness

    @property
    def population(self) -> np.ndarray:
        """The solution archive, ranked best-first (mirrors GeneticOptimizer.population)."""
        return self.archive
