"""Behavioral tests for the Simulated Annealing optimizer (src/annealing.py).

SA is a live backend in the Pareto tab but had no test. These lock its contracts on a
deterministic analytic objective (no model needed): it cools, the incumbent best never
regresses, the solution stays in bounds, and it makes progress toward the optimum.
"""
import numpy as np
import pytest

from src.annealing import SimulatedAnnealing

# Maximize -Σ(x-2)^2 → optimum at x = 2 in every dimension, fitness 0.
BOUNDS = [(-5.0, 5.0)] * 4


def _neg_quadratic(x):
    return -float(np.sum((np.asarray(x) - 2.0) ** 2))


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)


def test_temperature_cools_monotonically():
    sa = SimulatedAnnealing(_neg_quadratic, BOUNDS, initial_temp=100.0, cooling_rate=0.9)
    temps = [sa.step()["temperature"] for _ in range(15)]
    assert all(b < a for a, b in zip(temps, temps[1:])), "temperature must strictly cool"


def test_best_fitness_never_regresses():
    sa = SimulatedAnnealing(_neg_quadratic, BOUNDS, initial_temp=100.0, cooling_rate=0.9)
    bests = [sa.step()["best_fitness"] for _ in range(30)]
    # The incumbent best only updates on improvement, so it must be non-decreasing.
    assert all(b >= a - 1e-9 for a, b in zip(bests, bests[1:]))


def test_best_solution_stays_in_bounds():
    sa = SimulatedAnnealing(_neg_quadratic, BOUNDS, initial_temp=100.0, cooling_rate=0.85)
    for _ in range(40):
        sa.step()
    lo = np.array([b[0] for b in BOUNDS])
    hi = np.array([b[1] for b in BOUNDS])
    assert np.all(sa.best >= lo) and np.all(sa.best <= hi)


def test_converges_toward_optimum():
    sa = SimulatedAnnealing(_neg_quadratic, BOUNDS, initial_temp=200.0, cooling_rate=0.9)
    start = sa.best_fitness
    for _ in range(60):
        sa.step()
    # It should end meaningfully closer to the optimum (0) than it started, and land
    # in the neighborhood of x = 2 rather than an arbitrary corner.
    assert sa.best_fitness > start
    assert sa.best_fitness > -4.0, f"did not approach the optimum: {sa.best_fitness}"


def test_minimize_mode():
    # Same bowl, minimize Σ(x-2)^2 → optimum 0; best must be non-increasing.
    sa = SimulatedAnnealing(lambda x: -_neg_quadratic(x), BOUNDS,
                            initial_temp=200.0, cooling_rate=0.9, maximize=False)
    bests = [sa.step()["best_fitness"] for _ in range(40)]
    assert all(b <= a + 1e-9 for a, b in zip(bests, bests[1:]))
    assert sa.best_fitness < 4.0
