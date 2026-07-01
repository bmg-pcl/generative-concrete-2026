"""
Tests for physical-validity checks (src/physical.py) and their enforcement in the
generators (R2.2 / R2.3).
"""
import numpy as np
import pytest

from src.physical import (
    mix_volume, volume_error, repair_volume, enforce_volume, workability_flag,
    VOLUME_TOLERANCE, DENSITIES,
)
from src.data_fetcher import load_data
from src.generative_ga import PopulationInverseDesigner, PARAM_NAMES


def _to_arr(mix):
    return np.array([mix[p] for p in PARAM_NAMES], dtype=float)


def test_uci_rows_pass_volume_tolerance():
    """R2.2 density/tolerance validation: >=90% of real UCI mixes must be within tol."""
    df = load_data()
    errs = np.array([volume_error(dict(zip(PARAM_NAMES, r))) for r in df[PARAM_NAMES].values])
    assert float(np.mean(errs <= VOLUME_TOLERANCE)) >= 0.90


def test_corner_fails_and_repair_fixes_it():
    corner = dict(zip(PARAM_NAMES, [540, 360, 200, 120, 30, 1150, 1000, 28]))
    assert volume_error(corner) > VOLUME_TOLERANCE
    repaired = repair_volume(corner)
    assert volume_error(repaired) <= 1e-6
    assert repaired["age"] == 28  # non-material keys untouched


def test_enforce_volume_is_noop_when_valid():
    df = load_data()
    good = dict(zip(PARAM_NAMES, df[PARAM_NAMES].values[0]))
    # A real row is within tol → returned unchanged.
    assert enforce_volume(good) == good


def test_designer_cloud_is_balanced():
    """R2.2 gate: the designer sample cloud satisfies the volume balance."""
    d = PopulationInverseDesigner()
    samples = d.sample(45.0, n_samples=300, age=28.0)
    errs = np.array([volume_error(dict(zip(PARAM_NAMES, s))) for s in samples])
    assert float(np.mean(errs <= VOLUME_TOLERANCE)) >= 0.90


def test_best_mix_is_balanced():
    d = PopulationInverseDesigner()
    bm = d.best_mix(45.0, age=28.0)
    assert volume_error(bm) <= VOLUME_TOLERANCE


def test_workability_flag_boundaries():
    # Low w/b, no SP → warns.
    assert workability_flag(dict(cement=500, slag=0, ash=0, water=150, superplasticizer=0)) is not None
    # High w/b → warns.
    assert workability_flag(dict(cement=200, slag=0, ash=0, water=160, superplasticizer=0)) is not None
    # Normal → no warning.
    assert workability_flag(dict(cement=350, slag=0, ash=0, water=175, superplasticizer=8)) is None
