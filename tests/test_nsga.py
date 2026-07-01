"""
Tests for the NSGA-II / NSGA-III multi-objective optimizer (src/nsga.py).

Skipped automatically when pymoo is not installed. Small runs (few generations)
exercise the plumbing and the key guarantee: the returned front is non-dominated.
"""
import numpy as np
import pytest

pytest.importorskip("pymoo")

from src.nsga import run_nsga  # noqa: E402
from src.models import StrengthPredictor  # noqa: E402
from src.generative_ga import PARAM_NAMES, data_envelope  # noqa: E402
from src.ui_logic import pareto_front_mask  # noqa: E402


@pytest.fixture(scope="module")
def predictor():
    return StrengthPredictor()


def test_nsga2_returns_nondominated_front(predictor):
    out = run_nsga(predictor, algorithm="nsga2", pop_size=40, n_gen=15)
    assert out["algorithm"] == "NSGA-II"
    assert out["front_size"] >= 2
    assert out["mixes"].shape[1] == len(PARAM_NAMES)
    # Every returned point must be non-dominated among the returned set.
    mask = pareto_front_mask(out["strength"], out["carbon"], out["cost"])
    assert mask.all(), f"{(~mask).sum()} dominated points in the returned front"


def test_nsga_front_within_envelope(predictor):
    env = data_envelope()
    out = run_nsga(predictor, algorithm="nsga2", pop_size=40, n_gen=10)
    assert (out["mixes"] >= env[:, 0] - 1e-6).all()
    assert (out["mixes"] <= env[:, 1] + 1e-6).all()


def test_nsga3_runs(predictor):
    out = run_nsga(predictor, algorithm="nsga3", pop_size=60, n_gen=10, n_partitions=8)
    assert out["algorithm"] == "NSGA-III"
    assert out["front_size"] >= 2


def test_nsga_fixed_age(predictor):
    """R2.1 gate: NSGA holds a fixed design age across the whole front."""
    out = run_nsga(predictor, algorithm="nsga2", pop_size=40, n_gen=12, age=28.0)
    age_idx = PARAM_NAMES.index("age")
    assert np.allclose(out["mixes"][:, age_idx], 28.0)


def test_robust_nsga_front_in_support(predictor):
    """R1.3 gate: with robust=True, the whole Pareto front is in-support by construction."""
    out = run_nsga(predictor, algorithm="nsga2", pop_size=40, n_gen=20, robust=True)
    assert out["front_size"] >= 1
    assert predictor.in_support(out["mixes"]).all()


def test_warm_start_accepted(predictor):
    # A seed population smaller than pop_size must be padded and accepted.
    seed = np.tile(np.array([350, 100, 0, 175, 5, 1000, 750, 28], float), (10, 1))
    out = run_nsga(predictor, algorithm="nsga2", pop_size=40, n_gen=8, seed_population=seed)
    assert out["front_size"] >= 1
    assert len(out["history"]["best_strength"]) == 8
