"""
Tests for physical-validity checks (src/physical.py) and their enforcement in the
generators (R2.2 / R2.3).

R7.5 WP-2: mix_volume gains an optional `exotic` argument so exotic admixture mass
is no longer invisible to the volume balance. See
docs/specs/R7.5-chemistry-remediation.md WP-2 for the finding (150 kg/m3 of calcined
clay alone displaces ~0.058 m3, more than the entire VOLUME_TOLERANCE below).
"""
import numpy as np
import pytest

from src.physical import (
    mix_volume, volume_error, repair_volume, enforce_volume, workability_flag,
    VOLUME_TOLERANCE,
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


def test_mix_volume_default_arg_is_bit_identical_to_core_only_behaviour():
    """R7.5 WP-2: mix_volume(mix) with no second argument must be bit-identical to
    its pre-WP-2 value -- every existing caller (enforce_volume, the generators)
    passes only one argument and must be unaffected."""
    df = load_data()
    row = dict(zip(PARAM_NAMES, df[PARAM_NAMES].values[0]))
    v_no_arg = mix_volume(row)
    v_explicit_none = mix_volume(row, None)
    v_empty_exotic = mix_volume(row, {})
    assert v_no_arg == v_explicit_none == v_empty_exotic
    # And it matches the historical formula directly (core densities + air only).
    from src.physical import DENSITIES, AIR_FRACTION
    expected = sum(row.get(k, 0.0) / rho for k, rho in DENSITIES.items()) + AIR_FRACTION
    assert v_no_arg == expected


def test_mix_volume_accounts_for_exotic_dosing():
    """R7.5 WP-2 acceptance gate: 150 kg/m3 of calcined clay (density 2600 kg/m3)
    must add ~0.058 m3 (150/2600) on top of the base volume, and must be dosed
    through the `exotic` argument only -- the base-only call is untouched."""
    df = load_data()
    base = dict(zip(PARAM_NAMES, df[PARAM_NAMES].values[0]))
    v_base = mix_volume(base)
    v_with_clay = mix_volume(base, {"calcined_clay": 150})
    delta = v_with_clay - v_base
    assert delta == pytest.approx(150 / 2600, rel=1e-9)
    # Measured in the spec: this alone exceeds the entire VOLUME_TOLERANCE.
    assert delta > VOLUME_TOLERANCE


def test_mix_volume_exotic_sums_multiple_materials():
    df = load_data()
    base = dict(zip(PARAM_NAMES, df[PARAM_NAMES].values[0]))
    v_base = mix_volume(base)
    exotic = {"calcined_clay": 150, "steel_fiber": 80}  # 2600, 7850 kg/m3
    delta = mix_volume(base, exotic) - v_base
    assert delta == pytest.approx(150 / 2600 + 80 / 7850, rel=1e-9)


def test_workability_flag_boundaries():
    # Low w/b, no SP → warns.
    assert workability_flag(dict(cement=500, slag=0, ash=0, water=150, superplasticizer=0)) is not None
    # High w/b → warns.
    assert workability_flag(dict(cement=200, slag=0, ash=0, water=160, superplasticizer=0)) is not None
    # Normal → no warning.
    assert workability_flag(dict(cement=350, slag=0, ash=0, water=175, superplasticizer=8)) is None
