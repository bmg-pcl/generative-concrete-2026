"""
Tests for prediction intervals and out-of-support scoring (src/models.py).
"""
import numpy as np
import pytest

from src.models import StrengthPredictor
from src.generative_ga import PARAM_NAMES

# Near the dataset mean → in support; all-envelope-corners → far out of support.
IN_DIST = np.array([281, 73, 54, 181, 6, 972, 773, 45], dtype=float)
OUT_DIST = np.array([540, 360, 200, 120, 30, 1150, 1000, 365], dtype=float)


@pytest.fixture(scope="module")
def predictor():
    return StrengthPredictor()


def test_predict_interval_ordered_and_contains_signal(predictor):
    X = np.vstack([IN_DIST, OUT_DIST])
    lo, med, hi = predictor.predict_interval(X)
    assert (lo <= med).all() and (med <= hi).all()
    assert (hi - lo > 0).all()  # non-degenerate interval


def test_predict_variance_is_half_interval(predictor):
    lo, _, hi = predictor.predict_interval(IN_DIST)
    v = predictor.predict_variance(IN_DIST)
    assert v == pytest.approx((hi[0] - lo[0]) / 2.0, rel=1e-6)
    assert v > 0


def test_novelty_separates_in_and_out_of_support(predictor):
    n_in = float(predictor.novelty(IN_DIST)[0])
    n_out = float(predictor.novelty(OUT_DIST)[0])
    assert n_out > n_in
    assert predictor.in_support(IN_DIST)[0]
    assert not predictor.in_support(OUT_DIST)[0]


def test_novelty_batch_shape(predictor):
    X = np.vstack([IN_DIST, OUT_DIST, IN_DIST])
    nov = predictor.novelty(X)
    assert nov.shape == (3,)
    assert (nov >= 0).all()
