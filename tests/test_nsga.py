"""
Tests for the NSGA-II / NSGA-III multi-objective optimizer (src/nsga.py).

Skipped automatically when pymoo is not installed. Small runs (few generations)
exercise the plumbing and the key guarantee: the returned front is non-dominated.
"""
import numpy as np
import pytest

pytest.importorskip("pymoo")

from src.nsga import run_nsga, MixDesignProblem  # noqa: E402
from src.models import StrengthPredictor  # noqa: E402
from src.generative_ga import PARAM_NAMES, data_envelope  # noqa: E402
from src.ui_logic import pareto_front_mask, mix_dict, compute_metrics  # noqa: E402
from src.compliance import load_packs, check_compliance  # noqa: E402


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


def test_nsga_front_is_volume_balanced(predictor):
    """R2.2 gate: the NSGA front is batchable by construction (volume constraint)."""
    from src.physical import volume_error, VOLUME_TOLERANCE
    out = run_nsga(predictor, algorithm="nsga2", pop_size=40, n_gen=20)
    errs = np.array([volume_error(dict(zip(PARAM_NAMES, m))) for m in out["mixes"]])
    assert (errs <= VOLUME_TOLERANCE + 1e-6).all()


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


# --- WP-3b: optional compliance constraint (spec R8.2, WP-3 item 5) ---------
#
# Uses the `_fixture` pack (data/exposure_packs/_fixture.json, owned by WP-1)
# via `load_packs(include_hidden=True)`, same as tests/test_generative.py --
# see that file's header comment for why (its "A1" class has every rule
# present with real numeric values, so it can actually reach PASS).

def test_default_path_is_bit_identical_to_no_compliance(predictor):
    """compliance=None (the default) must produce the exact same front as
    omitting the argument entirely -- the new constraint code path must not
    execute at all. (Independently verified against pre-change nsga.py from
    git HEAD under the same seed -- see the WP-3b report -- this is the
    in-repo regression guard for that finding.)"""
    out1 = run_nsga(predictor, algorithm="nsga2", pop_size=30, n_gen=10, random_seed=3)
    out2 = run_nsga(predictor, algorithm="nsga2", pop_size=30, n_gen=10, random_seed=3,
                     compliance=None)
    assert np.array_equal(out1["mixes"], out2["mixes"])
    assert np.array_equal(out1["strength"], out2["strength"])
    assert np.array_equal(out1["carbon"], out2["carbon"])
    assert np.array_equal(out1["cost"], out2["cost"])
    assert out1["history"] == out2["history"]
    assert out2["compliance"] is None


def test_nsga_compliance_constraint_front_passes_requested_class(predictor):
    """Gate: a constrained NSGA run's front members pass the requested class,
    verified independently with check_compliance (not just trusted from the
    optimizer's own constraint bookkeeping)."""
    out = run_nsga(predictor, algorithm="nsga2", pop_size=40, n_gen=20,
                    compliance=("_fixture", "A1"))
    assert out["compliance"] is not None
    assert out["compliance"]["strength_basis"] == "conformal_lower_bound"
    assert out["compliance"]["all_pass"], "front reported non-compliant members"
    pack = load_packs(include_hidden=True)["_fixture"]
    strength_lo, _, _ = predictor.predict_interval(out["mixes"])
    for x, slo in zip(out["mixes"], strength_lo):
        result = check_compliance(mix_dict(x), pack, "A1", strength_lo=float(slo))
        assert result["verdict"] == "PASS"


def test_nsga_compliance_constraint_keeps_three_objectives(predictor):
    """Gate: compliance must be a pymoo CONSTRAINT, not a 4th objective -- the
    front's objective dimensionality (strength/carbon/cost) must be unchanged."""
    out = run_nsga(predictor, algorithm="nsga2", pop_size=40, n_gen=15,
                    compliance=("_fixture", "A1"))
    assert out["mixes"].shape[1] == len(PARAM_NAMES)
    # Exactly 3 objective-derived arrays are returned, matching the unconstrained
    # front's shape -- no 4th "compliance objective" array exists anywhere.
    assert out["strength"].shape == out["carbon"].shape == out["cost"].shape
    assert out["strength"].ndim == 1
    baseline = run_nsga(predictor, algorithm="nsga2", pop_size=40, n_gen=15)
    assert out["mixes"].shape[1] == baseline["mixes"].shape[1]


# --- R8.5 P1: the coherence contract, kept forever (NSGA leg) ---------------------
#
# See tests/test_ui_logic.py's matching block for the full rationale. This is
# the "Same for the NSGA objective column" half of the spec's gate: the front's
# carbon OBJECTIVE COLUMN (MixDesignProblem._evaluate's out["F"][:, 1]) must
# equal compute_metrics's displayed carbon for the identical mix/config. Tested
# directly against MixDesignProblem._evaluate (not a full run_nsga optimization)
# so the coherence gate is cheap and deterministic -- it is arithmetic
# reconciliation, not a search-quality property.

def _p1_factors_variants():
    from src.chemistry_simple import CARBON_FACTORS
    overridden = dict(CARBON_FACTORS)
    overridden["cement"] = overridden["cement"] * 0.5
    return (None, overridden)


_P1_NSGA_AXES = [
    (advanced, transport_km, transport_detail, factors_override, clinker_source)
    for advanced in (False, True)
    for transport_km in (0.0, 500.0)
    for transport_detail in (False, True)
    for factors_override in _p1_factors_variants()
    for clinker_source in (None, {"kiln_fuel": "natural_gas", "electricity": "hydro"})
]


@pytest.mark.parametrize(
    "advanced,transport_km,transport_detail,factors_override,clinker_source", _P1_NSGA_AXES,
    ids=[f"advanced={a}-km={k}-detail={t}-factors={'override' if f else 'default'}-"
         f"clinker={'set' if c else 'none'}"
         for a, k, t, f, c in _P1_NSGA_AXES],
)
def test_p1_nsga_objective_carbon_column_matches_compute_metrics(
    predictor, advanced, transport_km, transport_detail, factors_override, clinker_source,
):
    mix = np.array([350.0, 100.0, 0.0, 175.0, 5.0, 1000.0, 750.0, 28.0])
    carbon_kwargs = {
        "transport_km": transport_km,
        "cement_type": "OPC",
        "factors": factors_override,
        "clinker_source": clinker_source,
        "transport_detail": transport_detail,
    }
    bounds = data_envelope()
    problem = MixDesignProblem(predictor, bounds, advanced, costs=None,
                               carbon_kwargs=carbon_kwargs)
    out = {}
    problem._evaluate(mix.reshape(1, -1), out)
    nsga_carbon = float(out["F"][0, 1])

    m = compute_metrics(mix, {}, {}, predictor, advanced=advanced, carbon_kwargs=carbon_kwargs)

    assert nsga_carbon == pytest.approx(m["carbon"]), (
        f"NSGA carbon objective column diverged from compute_metrics carbon: "
        f"advanced={advanced} transport_km={transport_km} "
        f"transport_detail={transport_detail} clinker_source={clinker_source}"
    )


# --- R8.5 P2: robust carbon -- optimize the upper bound, symmetric with robust strength --

def test_p2_nsga_robust_carbon_default_bit_identical(predictor):
    """No caller touches `robust_carbon` (default False) -> identical front,
    same seed."""
    out1 = run_nsga(predictor, algorithm="nsga2", pop_size=30, n_gen=10, random_seed=3)
    out2 = run_nsga(predictor, algorithm="nsga2", pop_size=30, n_gen=10, random_seed=3,
                    robust_carbon=False)
    assert np.array_equal(out1["mixes"], out2["mixes"])
    assert np.array_equal(out1["carbon"], out2["carbon"])
    assert out1["carbon_basis"] == out2["carbon_basis"] == "point"


def test_p2_nsga_robust_carbon_swaps_column_dimensionality_unchanged(predictor):
    """Gate: robust_carbon swaps the carbon column -- front stays 3-objective
    (strength/carbon/cost), and the basis is disclosed correctly either way."""
    baseline = run_nsga(predictor, algorithm="nsga2", pop_size=40, n_gen=15)
    assert baseline["carbon_basis"] == "point"
    robust = run_nsga(predictor, algorithm="nsga2", pop_size=40, n_gen=15, robust_carbon=True)
    assert robust["carbon_basis"] == "upper_95"
    assert robust["mixes"].shape[1] == baseline["mixes"].shape[1] == len(PARAM_NAMES)
    assert robust["strength"].shape == robust["carbon"].shape == robust["cost"].shape
    assert robust["strength"].ndim == 1


def test_p2_nsga_robust_carbon_zero_uncertainty_bit_identical(predictor, monkeypatch):
    """The MixDesignProblem leg of the zero-uncertainty gate: with every
    registry uncertainty patched to zero, the robust_carbon objective column
    equals the point column exactly."""
    import src.ui_logic as ui_logic
    monkeypatch.setattr(ui_logic, "factor_uncertainties_view", lambda: {})
    mix = np.array([350.0, 100.0, 0.0, 175.0, 5.0, 1000.0, 750.0, 28.0])
    bounds = data_envelope()
    carbon_kwargs = {"transport_km": 150.0}
    point_problem = MixDesignProblem(predictor, bounds, False, costs=None,
                                     carbon_kwargs=carbon_kwargs, robust_carbon=False)
    robust_problem = MixDesignProblem(predictor, bounds, False, costs=None,
                                      carbon_kwargs=carbon_kwargs, robust_carbon=True)
    out_point, out_robust = {}, {}
    point_problem._evaluate(mix.reshape(1, -1), out_point)
    robust_problem._evaluate(mix.reshape(1, -1), out_robust)
    assert out_point["F"][0, 1] == pytest.approx(out_robust["F"][0, 1])


def test_nsga_compliance_uses_lower_bound_not_point_estimate(predictor):
    """The constraint must be built on strength_lo, not the mean -- confirm the
    reported front verdicts were computed against predict_interval's lower
    bound, not predictor.predict_batch's point estimate (which can disagree)."""
    out = run_nsga(predictor, algorithm="nsga2", pop_size=40, n_gen=15,
                    compliance=("_fixture", "A1"))
    for r in out["compliance"]["results"]:
        strength_rule = next(x for x in r["rules"] if x["rule"] == "min_strength_MPa")
        assert strength_rule["basis"] == "conformal_lower_bound (strength_lo supplied)"
