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


def test_interval_coverage(predictor):
    """R1.1 gate: the conformalised 90% interval must actually cover ~90% of held-out
    points. Recompute the same 80/20 split used in train() and measure on the test set."""
    from sklearn.model_selection import train_test_split
    from src.data_fetcher import load_data
    df = load_data()
    X = df.drop("strength", axis=1)
    y = df["strength"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lo, _, hi = predictor.predict_interval(np.asarray(X_test, dtype=float))
    yt = np.asarray(y_test, dtype=float)
    coverage = float(np.mean((yt >= lo) & (yt <= hi)))
    assert 0.84 <= coverage <= 0.97, f"90% interval coverage was {coverage:.3f}"


def test_novelty_threshold_is_calibrated(predictor):
    """R1.2 gate: the stored threshold is data-derived (not the 1.5 fallback), and most
    genuine (held-out) mixes are in-support under it."""
    from sklearn.model_selection import train_test_split
    from src.data_fetcher import load_data
    df = load_data()
    X = df.drop("strength", axis=1)
    y = df["strength"]
    _, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    thr = predictor.support_threshold()
    assert thr != 1.5  # calibrated, not the fallback
    frac_in = float(np.mean(predictor.in_support(np.asarray(X_test, dtype=float))))
    assert frac_in >= 0.90


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
