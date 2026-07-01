"""
Regression tests for the Phase 3/4 generative fixes (see docs/FIX_PLAN.md).

These lock in the behaviours that were previously broken:
  * the sampler must depend on the target (it used to return an identical cloud);
  * generated mixes must stay inside the training-data envelope (it used to emit
    water below the dataset minimum);
  * the inverse planner must actually reach the requested strength;
  * physics.py must be gone (it was dead + broken);
  * evaluate_uncertainty must be deterministic (it used to be np.random.uniform).
"""
import importlib

import numpy as np
import pytest

from src.generative_ga import PopulationInverseDesigner, data_envelope, PARAM_NAMES
from src.bayesian import BayesFlowExplorer
from src.chemistry_advanced import inverse_plan_mix


@pytest.fixture(scope="module")
def designer():
    return PopulationInverseDesigner()


def test_designer_hits_target(designer):
    """The best candidate should predict close to the requested strength."""
    for target in (25.0, 45.0, 65.0):
        mixes, errors = designer.design(target, generations=40)
        achieved = designer.predictor.predict(mixes[0])
        assert abs(achieved - target) < 3.0, f"target {target}: got {achieved:.1f} MPa"


def test_samples_stay_in_envelope(designer):
    """No generated sample may fall outside the per-parameter data envelope."""
    env = data_envelope()
    samples = designer.sample(45.0, n_samples=500)
    assert samples.shape == (500, len(PARAM_NAMES))
    assert (samples >= env[:, 0] - 1e-6).all()
    assert (samples <= env[:, 1] + 1e-6).all()


def test_sample_posterior_is_target_conditioned():
    """Different targets must yield different clouds (the old sampler did not)."""
    explorer = BayesFlowExplorer()
    lo = explorer.sample_posterior(25.0, n_samples=300)
    hi = explorer.sample_posterior(70.0, n_samples=300)
    lo_mean = np.array([explorer.predictor.predict(s) for s in lo]).mean()
    hi_mean = np.array([explorer.predictor.predict(s) for s in hi]).mean()
    assert hi_mean - lo_mean > 15.0
    assert not np.allclose(lo.mean(axis=0), hi.mean(axis=0))


def test_evaluate_uncertainty_is_deterministic():
    explorer = BayesFlowExplorer()
    mix = np.array([120, 0, 0, 140, 0, 1000, 750, 28], dtype=float)
    assert explorer.evaluate_uncertainty(mix) == explorer.evaluate_uncertainty(mix)


def test_inverse_plan_mix_reaches_target():
    """The planner should reach its target and stay in-distribution."""
    env = dict(zip(PARAM_NAMES, data_envelope()))
    designer = PopulationInverseDesigner()
    mix = inverse_plan_mix(45.0, target_carbon_kg=250)
    achieved = designer.predictor.predict(np.array([mix[k] for k in PARAM_NAMES], dtype=float))
    assert abs(achieved - 45.0) < 4.0
    # Water must not drop below the dataset minimum (the old bug).
    assert mix["water"] >= env["water"][0] - 1e-6


def test_physics_module_removed():
    """physics.py was dead + broken; it must no longer be importable."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("src.physics")
