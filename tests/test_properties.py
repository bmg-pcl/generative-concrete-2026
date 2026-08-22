"""
Tests for the PropertyModel abstraction and the slump model (src/properties.py,
R8.1). Mirrors tests/test_models.py's structure (StrengthPredictor, R7.1) but for
the SECOND measured property, whose corpus is two orders of magnitude smaller.
"""
import numpy as np
import pytest

from src.data_fetcher import load_slump_data
from src.models import StrengthPredictor
from src.properties import (
    SLUMP_FEATURES,
    SLUMP_MODEL_PATH,
    PropertyModel,
    get_slump_model,
    slump_data_envelope,
    slump_estimate,
)

# A mix comfortably inside the slump corpus's own envelope (see
# slump_data_envelope()): cement/slag/ash/water/SP/coarse/fine.
IN_DIST = np.array([250.0, 80.0, 100.0, 190.0, 8.0, 900.0, 750.0])
# A mix far outside it in every dimension.
OUT_DIST = np.array([900.0, 400.0, 400.0, 500.0, 60.0, 2000.0, 2000.0])

# Strength-in-support / slump-out-of-support exhibit (R8.1 acceptance gate #3): an
# ordinary mix with SP=0. The strength corpus (1030 rows) ranges SP 0-32.2, so SP=0
# is unremarkable there. The slump corpus (103 rows) ranges SP 4.4-19 -- every row
# used a superplasticizer -- so SP=0 sits off the edge of that specific envelope even
# though nothing else about the mix is unusual.
DIVERGENT_MIX = {
    "cement": 280.0, "slag": 0.0, "ash": 0.0, "water": 185.0,
    "superplasticizer": 0.0, "coarse_agg": 968.0, "fine_agg": 780.0, "age": 28.0,
}


@pytest.fixture(scope="module")
def slump_model():
    return get_slump_model()


# -- loader --------------------------------------------------------------

def test_load_slump_data_shape_and_columns():
    df = load_slump_data()
    assert len(df) == 103
    assert list(df.columns) == [
        "cement", "slag", "ash", "water", "superplasticizer", "coarse_agg",
        "fine_agg", "slump_cm", "flow_cm", "strength",
    ]
    assert df.isnull().sum().sum() == 0


def test_load_slump_data_strength_column_loaded_but_unused():
    """The corpus's own Compressive Strength column is loaded for inspection but must
    never be a model feature (docs/specs/R8.1: 'load it, ignore it')."""
    df = load_slump_data()
    assert "strength" in df.columns
    assert "strength" not in SLUMP_FEATURES


def test_slump_features_are_seven_materials_no_age():
    """docs/specs/R8.1 point 3: slump is measured on fresh concrete, so the model's
    inputs are PARAM_NAMES[:7] (materials only) -- no invented 'age' feature."""
    assert SLUMP_FEATURES == [
        "cement", "slag", "ash", "water", "superplasticizer", "coarse_agg", "fine_agg",
    ]
    assert "age" not in SLUMP_FEATURES


# -- PropertyModel: coverage / width / coherence --------------------------

def test_measured_coverage_in_loose_gate(slump_model):
    """103 rows cannot pin coverage tightly -- the gate is deliberately loose
    ([0.80, 0.98]); report whatever is measured, never tune toward 0.90."""
    cov = slump_model.held_out_["coverage"]
    assert 0.80 <= cov <= 0.98, f"measured held-out coverage was {cov:.3f}"


def test_median_interval_width_is_reported(slump_model):
    """Not a target to minimise -- just confirm it is a real, finite, positive,
    reported number (a small corpus should give a WIDE interval)."""
    width = slump_model.held_out_["median_interval_width"]
    assert np.isfinite(width)
    assert width > 0


def test_predict_interval_ordered_and_nondegenerate(slump_model):
    X = np.vstack([IN_DIST, OUT_DIST])
    lo, med, hi = slump_model.predict_interval(X)
    assert (lo <= med).all() and (med <= hi).all()
    assert (hi - lo > 0).all()


def test_coherence_sweep_lo_le_point_le_hi(slump_model):
    """R7.1's coherence gate, ported: sweep 1000 points over the slump corpus's OWN
    envelope and assert the point estimate never falls outside its own interval."""
    env = slump_data_envelope()
    rng = np.random.default_rng(0)
    grid = rng.uniform(env[:, 0], env[:, 1], size=(1000, len(SLUMP_FEATURES)))
    lo, med, hi = slump_model.predict_interval(grid)
    assert np.array_equal(med, slump_model.predict(grid))
    assert (med >= lo).all(), f"{int(np.sum(med < lo))} points below their lower bound"
    assert (med <= hi).all(), f"{int(np.sum(med > hi))} points above their upper bound"


# -- novelty / support gate ------------------------------------------------

def test_novelty_separates_in_and_out_of_support(slump_model):
    n_in = float(slump_model.novelty(IN_DIST)[0])
    n_out = float(slump_model.novelty(OUT_DIST)[0])
    assert n_out > n_in
    assert slump_model.in_support(IN_DIST)[0]
    assert not slump_model.in_support(OUT_DIST)[0]


def test_novelty_threshold_is_calibrated(slump_model):
    assert slump_model.support_threshold() != 1.5


def test_strength_in_support_slump_out_of_support():
    """R8.1 acceptance gate: the slump support gate is genuinely distinct from the
    strength gate. DIVERGENT_MIX is an ordinary mix (SP=0) that the 1030-row strength
    corpus considers unremarkable but the 103-row slump corpus (every row SP >= 4.4)
    does not -- proving each property needs its OWN gate, not a shared one."""
    strength_model = StrengthPredictor()
    mix8 = np.array([DIVERGENT_MIX[k] for k in
                      ["cement", "slag", "ash", "water", "superplasticizer",
                       "coarse_agg", "fine_agg", "age"]], dtype=float)
    assert bool(strength_model.in_support(mix8)[0]), (
        "expected DIVERGENT_MIX to be in-support for strength"
    )

    slump_model = get_slump_model()
    mix7 = mix8[:7]
    assert not bool(slump_model.in_support(mix7)[0]), (
        "expected DIVERGENT_MIX to be OUT of support for slump"
    )


# -- save/load round-trip --------------------------------------------------

def test_save_load_round_trip(slump_model, tmp_path):
    path = str(tmp_path / "roundtrip_slump_model.json")
    slump_model.save(path)
    reloaded = PropertyModel.load(path)

    env = slump_data_envelope()
    rng = np.random.default_rng(1)
    grid = rng.uniform(env[:, 0], env[:, 1], size=(200, len(SLUMP_FEATURES)))

    lo1, med1, hi1 = slump_model.predict_interval(grid)
    lo2, med2, hi2 = reloaded.predict_interval(grid)
    assert np.allclose(lo1, lo2)
    assert np.allclose(med1, med2)
    assert np.allclose(hi1, hi2)
    assert np.array_equal(slump_model.novelty(grid), reloaded.novelty(grid))


def test_committed_model_artifact_exists_and_loads():
    import os
    assert os.path.exists(SLUMP_MODEL_PATH), (
        f"{SLUMP_MODEL_PATH} must be committed, same convention as "
        "models/strength_quantiles.json"
    )
    m = PropertyModel.load(SLUMP_MODEL_PATH)
    assert m.feature_names == SLUMP_FEATURES


# -- slump_estimate() honest degradation -----------------------------------

def test_slump_estimate_in_support_returns_model_basis():
    mix = dict(zip(SLUMP_FEATURES, IN_DIST.tolist()))
    result = slump_estimate(mix)
    assert result["in_support"] is True
    assert result["basis"] == "model"
    assert result["slump_cm"] is not None
    assert result["lo"] <= result["slump_cm"] <= result["hi"]


def test_slump_estimate_out_of_support_returns_heuristic_basis_and_reason():
    mix = dict(zip(SLUMP_FEATURES, OUT_DIST.tolist()))
    result = slump_estimate(mix)
    assert result["in_support"] is False
    assert result["basis"] == "heuristic"
    assert result["slump_cm"] is None, "no confident model number outside the envelope"
    assert result["lo"] is None and result["hi"] is None
    assert result["reason"]  # non-empty


def test_slump_estimate_divergent_mix_is_heuristic():
    """The strength-in-support/slump-out-of-support mix must degrade honestly
    through the public entry point too, not just at the PropertyModel level."""
    result = slump_estimate(DIVERGENT_MIX)
    assert result["basis"] == "heuristic"
    assert result["slump_cm"] is None


# -- fit() guardrails -------------------------------------------------------

def test_fit_refuses_too_few_rows():
    with pytest.raises(ValueError):
        PropertyModel(name="tiny", feature_names=["a", "b"]).fit(
            np.random.rand(10, 2), np.random.rand(10)
        )
