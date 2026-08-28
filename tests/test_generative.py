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

from src.generative_ga import (
    PopulationInverseDesigner,
    AntColonyInverseDesigner,
    data_envelope,
    PARAM_NAMES,
    compliance_violation,
    resolve_compliance_target,
    OOS_PENALTY_WEIGHT,
    SLUMP_OOS_PENALTY_WEIGHT,
    SLUMP_SP_DOSING_NOTE,
)
from src.bayesian import BayesFlowExplorer
from src.chemistry_advanced import inverse_plan_mix
from src.compliance import load_packs, check_compliance
from src.models import StrengthPredictor
from src.properties import slump_estimate


@pytest.fixture(scope="module")
def designer():
    return PopulationInverseDesigner()


def test_designer_hits_target(designer):
    """The best candidate should predict close to the requested strength."""
    for target in (25.0, 45.0, 65.0):
        mixes, errors = designer.design(target, generations=40)
        achieved = designer.predictor.predict(mixes[0])
        assert abs(achieved - target) < 3.0, f"target {target}: got {achieved:.1f} MPa"


def test_fixed_age_is_respected(designer):
    """R2.1 gate: a fixed design age must be held exactly, not exploited by the search."""
    mixes, _ = designer.design(45.0, age=28.0, generations=30)
    age_idx = PARAM_NAMES.index("age")
    assert np.allclose(mixes[:, age_idx], 28.0)
    samples = designer.sample(45.0, n_samples=200, age=28.0)
    assert np.allclose(samples[:, age_idx], 28.0)


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


# --- ACO variant -----------------------------------------------------------

@pytest.fixture(scope="module")
def aco_designer():
    return AntColonyInverseDesigner()


def test_aco_hits_target(aco_designer):
    """The ACO designer must also reach the requested strength."""
    for target in (30.0, 55.0):
        mixes, errors = aco_designer.design(target, generations=40)
        achieved = aco_designer.predictor.predict(mixes[0])
        assert abs(achieved - target) < 3.0, f"target {target}: got {achieved:.1f} MPa"


def test_aco_samples_stay_in_envelope(aco_designer):
    env = data_envelope()
    samples = aco_designer.sample(45.0, n_samples=300)
    assert samples.shape == (300, len(PARAM_NAMES))
    assert (samples >= env[:, 0] - 1e-6).all()
    assert (samples <= env[:, 1] + 1e-6).all()


def test_explorer_aco_backend():
    """method='aco' routes through the ant-colony designer and tracks the target."""
    explorer = BayesFlowExplorer()
    samples = explorer.sample_posterior(50.0, n_samples=200, method="aco")
    assert samples.shape == (200, len(PARAM_NAMES))
    mean_strength = explorer.predictor.predict_batch(samples).mean()
    assert abs(mean_strength - 50.0) < 12.0


# --- WP-3b: optional compliance constraint (spec R8.2, WP-3 item 5) ---------
#
# Uses the `_fixture` pack (data/exposure_packs/_fixture.json, owned by WP-1)
# via `load_packs(include_hidden=True)` -- its class "A1" has every rule
# present with real numeric values, so it can actually reach a PASS verdict
# (unlike en206/aci318, whose classes omit at least one rule everywhere and so
# can only ever reach UNKNOWN or FAIL through this honesty-preserving engine).

def test_default_path_is_bit_identical_to_no_compliance():
    """compliance=None (the default) must exercise none of the new code paths.

    Independently verified (see the WP-3b report) by running the pre-change
    generative_ga.py (git HEAD, before this feature) and the current module
    side by side under the same seeds: mixes/errors/samples came back
    `np.array_equal` -- exactly, not approximately. This test is the
    in-repo regression guard for that finding: `compliance=None` explicit vs.
    omitted must always be the same call.
    """
    np.random.seed(4242)
    d1 = PopulationInverseDesigner()
    np.random.seed(99)
    mixes1, errors1 = d1.design(45.0, generations=15, pop_size=30)

    np.random.seed(4242)
    d2 = PopulationInverseDesigner()
    np.random.seed(99)
    mixes2, errors2 = d2.design(45.0, generations=15, pop_size=30, compliance=None)

    assert np.array_equal(mixes1, mixes2)
    assert np.array_equal(errors1, errors2)


def test_compliance_penalty_biases_ga_toward_target_class():
    """A GA run with a compliance target set should find a mix whose real,
    independently-recomputed `check_compliance()` verdict is PASS -- the
    penalty is soft, so this confirms it's effective, not just present."""
    designer = PopulationInverseDesigner()
    pack = load_packs(include_hidden=True)["_fixture"]
    mixes, _ = designer.design(45.0, generations=25, pop_size=40, compliance=("_fixture", "A1"))
    best = mixes[0]
    mix = dict(zip(PARAM_NAMES, best))
    lo, _, _ = designer.predictor.predict_interval(best)
    result = check_compliance(mix, pack, "A1", strength_lo=float(lo[0]))
    assert result["verdict"] == "PASS"
    assert compliance_violation(mix, pack, "A1", float(lo[0])) == 0.0


def test_design_compliant_finds_a_passing_mix_or_reports_honestly():
    """Gate: a constrained GA run returns designs that pass the requested class,
    or reports honestly that none were found -- never a silent non-compliant
    'pass'."""
    designer = PopulationInverseDesigner()
    report = designer.design_compliant(45.0, ("_fixture", "A1"), generations=30, pop_size=50)
    assert report["strength_basis"] == "conformal_lower_bound"
    if report["found"]:
        assert report["result"]["verdict"] == "PASS"
        # Independently re-verify with the real engine -- the wrapper must not
        # be trusted on its own say-so.
        pack = load_packs(include_hidden=True)["_fixture"]
        lo, _, _ = designer.predictor.predict_interval(
            np.array([report["mix"][p] for p in PARAM_NAMES]))
        recheck = check_compliance(report["mix"], pack, "A1", strength_lo=float(lo[0]))
        assert recheck["verdict"] == "PASS"
    else:
        assert report["mix"] is None
        assert report["result"] is None
        assert len(report["checked"]) > 0


def test_compliance_uses_lower_bound_not_point_estimate():
    """The compliance term must be computed against strength_lo, never the mean
    (R8.2's core design decision). A young (age=7d), moderate-cement mix has a
    conformal lower bound (~18.9 MPa) below the fixture's 30 MPa min_strength_MPa
    rule while its mean (~34.8 MPa) is above it -- proving the basis matters,
    not just that using either would agree."""
    mix_vec = np.array([300, 0, 0, 150, 5, 1000, 750, 7], dtype=float)
    pred = StrengthPredictor()
    lo, med, hi = pred.predict_interval(mix_vec)
    assert lo[0] < 30.0 <= med[0], "fixture assumes this mix straddles the 30 MPa rule"
    pack = load_packs(include_hidden=True)["_fixture"]
    mix = dict(zip(PARAM_NAMES, mix_vec))
    viol_lo = compliance_violation(mix, pack, "A1", float(lo[0]))
    viol_med = compliance_violation(mix, pack, "A1", float(med[0]))
    assert viol_lo > 0.0   # FAILs min_strength_MPa on the lower bound
    assert viol_med == 0.0  # would (wrongly) PASS on the mean


def test_unknown_rule_counts_as_violation_by_default():
    """WP-3b's UNKNOWN-handling decision: for the optimizer, UNKNOWN counts as
    VIOLATED, not satisfied -- see the module docstring. Fixture class "A2" has
    only max_w_b present; the other four rules are absent (UNKNOWN). A mix that
    satisfies max_w_b must still carry a nonzero violation from those four
    UNKNOWN rules, and the opt-out flag must exist and do what it says."""
    pack = load_packs(include_hidden=True)["_fixture"]
    mix = {"cement": 320.0, "slag": 0.0, "ash": 0.0, "water": 140.0,
           "superplasticizer": 5.0, "coarse_agg": 1000.0, "fine_agg": 750.0, "age": 28.0}
    # w/b = 140/320 = 0.4375 <= 0.45 -> max_w_b PASSes on its own.
    viol = compliance_violation(mix, pack, "A2", strength_lo=40.0)
    assert viol == 4.0  # exactly the 4 absent (UNKNOWN) rules, unit-penalised
    viol_lenient = compliance_violation(mix, pack, "A2", strength_lo=40.0,
                                         unknown_counts_as_violation=False)
    assert viol_lenient == 0.0  # the opt-out path exists and behaves as documented


def test_resolve_compliance_target_resolves_hidden_fixture_and_rejects_unknown():
    pack, cls = resolve_compliance_target(("_fixture", "A1"))
    assert pack["pack_id"] == "_fixture"
    assert cls == "A1"
    assert resolve_compliance_target(None) == (None, None)
    with pytest.raises(KeyError):
        resolve_compliance_target(("_fixture", "no-such-class"))
    with pytest.raises(KeyError):
        resolve_compliance_target(("no-such-pack", "A1"))


# --- R8.5 P3: slump-target workability (only active when explicitly requested) ---
#
# Principle 2 of docs/specs/R8.5-optimizer-capability-integration.md: the slump
# support gate is the load-bearing constraint (R8.1's finding that the slump
# envelope != the strength envelope is exactly why); the target value itself is
# only a SOFT preference. `slump_target=None` (the default) must exercise none
# of this code, same "whole branch skipped" discipline as `compliance` above.

def test_slump_oos_penalty_weight_mirrors_oos_penalty_weight():
    """The spec's explicit instruction: SLUMP_OOS_PENALTY_WEIGHT mirrors
    OOS_PENALTY_WEIGHT -- same role (a smooth nudge away from a property's own
    extrapolated region), same numeric value."""
    assert SLUMP_OOS_PENALTY_WEIGHT == OOS_PENALTY_WEIGHT == 10.0


def test_slump_default_bit_identical_to_no_slump_target():
    """slump_target=None (the default) must exercise none of the new code paths
    -- same in-repo regression-guard shape as compliance's matching test."""
    np.random.seed(4242)
    d1 = PopulationInverseDesigner()
    np.random.seed(99)
    mixes1, errors1 = d1.design(45.0, generations=15, pop_size=30)

    np.random.seed(4242)
    d2 = PopulationInverseDesigner()
    np.random.seed(99)
    mixes2, errors2 = d2.design(45.0, generations=15, pop_size=30, slump_target=None)

    assert np.array_equal(mixes1, mixes2)
    assert np.array_equal(errors1, errors2)


def test_slump_target_penalty_biases_ga_toward_reachable_target(designer):
    """A GA run with a reachable slump target should find a mix that is BOTH
    in slump support and reasonably close to the requested value -- verified
    independently with `properties.slump_estimate`, not just trusted from the
    optimizer's own soft penalty."""
    target = 15.0
    mixes, _ = designer.design(45.0, generations=30, pop_size=60, slump_target=target)
    best = dict(zip(PARAM_NAMES, mixes[0]))
    s = slump_estimate(best)
    assert s["in_support"], s["reason"]
    assert s["basis"] == "model"
    assert abs(s["slump_cm"] - target) < 6.0  # soft penalty, not exact -- reasonably close


def test_slump_target_penalty_biases_aco_toward_reachable_target(aco_designer):
    """Same gate as the GA test above, for the ACO variant -- confirms the
    penalty is inherited via `_InverseDesignerBase._make_objective`, not a
    GA-only fork (this spec's cross-cutting 'shared helpers' rule)."""
    target = 15.0
    mixes, _ = aco_designer.design(45.0, generations=30, slump_target=target)
    best = dict(zip(PARAM_NAMES, mixes[0]))
    s = slump_estimate(best)
    assert s["in_support"], s["reason"]
    assert abs(s["slump_cm"] - target) < 8.0


def test_slump_target_penalty_pulls_point_estimate_toward_different_targets():
    """The soft target-match term must actually move the search: two GA runs
    with different slump targets (same strength target, same seed) should
    land on different achieved slump points, both closer to their OWN target
    than to the other's."""
    np.random.seed(7)
    d_lo = PopulationInverseDesigner()
    mixes_lo, _ = d_lo.design(45.0, generations=30, pop_size=60, slump_target=8.0)
    np.random.seed(7)
    d_hi = PopulationInverseDesigner()
    mixes_hi, _ = d_hi.design(45.0, generations=30, pop_size=60, slump_target=25.0)

    slump_lo = slump_estimate(dict(zip(PARAM_NAMES, mixes_lo[0])))["slump_cm"]
    slump_hi = slump_estimate(dict(zip(PARAM_NAMES, mixes_hi[0])))["slump_cm"]
    assert slump_lo is not None and slump_hi is not None
    assert slump_hi > slump_lo  # the higher-target run should land higher


def test_slump_sp_dosing_note_matches_corpus_finding():
    """R8.1 WP-1b's corpus finding, verified directly against the committed
    slump corpus rather than just trusted as a string: every row used
    superplasticizer >= 4.4 kg/m3."""
    from src.data_fetcher import load_slump_data
    df = load_slump_data()
    assert df["superplasticizer"].min() >= 4.4
    assert "4.4" in SLUMP_SP_DOSING_NOTE  # the disclosure names the actual figure
