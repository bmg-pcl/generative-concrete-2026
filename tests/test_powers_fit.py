"""
Tests for WP-F phase 1 -- Powers gel-space-ratio calibration
(see docs/specs/R8.0-carbon-chemistry-deepening.md, "WP-F").

This is an evidence-artifact package with a fence: `scripts/fit_powers.py`
fits exactly two numbers (Powers' A and n) against the existing, uncalibrated
`hydration_kinetics` model and writes `models/powers_calibration.json`. These
tests gate:
  - the JSON schema (keys present, types correct) -- not any literature match;
  - determinism (two runs -> byte-identical JSON);
  - n asserted only as finite and positive (the [1.5, 4.5] sanity band is
    reported, never a pass bar -- WP-F is explicit that a weak fit / an
    off-canonical n is a valid finding, not a failure);
  - that the script does not import xgboost, `src.models`, or `ui/` -- WP-F
    is zero runtime integration.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "fit_powers.py"
JSON_PATH = REPO_ROOT / "models" / "powers_calibration.json"

sys.path.insert(0, str(REPO_ROOT))


def _run_script() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.fixture(scope="module")
def result_json():
    proc = _run_script()
    assert proc.returncode == 0, f"fit_powers.py failed:\n{proc.stdout}\n{proc.stderr}"
    with open(JSON_PATH) as f:
        return json.load(f)


def test_script_runs_and_writes_json(result_json):
    assert JSON_PATH.exists()


def test_stdout_contains_markdown_report():
    proc = _run_script()
    assert proc.returncode == 0
    assert "# WP-F Phase 1" in proc.stdout
    assert "R^2" in proc.stdout
    assert "Interpretation" in proc.stdout


def test_determinism_two_runs_byte_identical():
    proc1 = _run_script()
    assert proc1.returncode == 0
    bytes1 = JSON_PATH.read_bytes()

    proc2 = _run_script()
    assert proc2.returncode == 0
    bytes2 = JSON_PATH.read_bytes()

    assert bytes1 == bytes2, "two runs of fit_powers.py must produce byte-identical JSON"
    assert proc1.stdout == proc2.stdout, "stdout report must also be deterministic"


# --------------------------------------------------------------------------
# Schema: required keys present, correct types. Not a literature-match gate.
# --------------------------------------------------------------------------

def test_schema_top_level_keys(result_json):
    for key in (
        "description",
        "formulas",
        "assumptions",
        "fit",
        "rows",
        "age_bucket_residuals_mpa",
        "scm_fraction_bucket_residuals_mpa",
    ):
        assert key in result_json, f"missing top-level key: {key}"


def test_schema_formulas(result_json):
    formulas = result_json["formulas"]
    for key in ("gel_space_ratio", "powers_law", "log_linear_fit"):
        assert key in formulas
        assert isinstance(formulas[key], str)
        assert len(formulas[key]) > 0


def test_schema_assumptions(result_json):
    assumptions = result_json["assumptions"]
    assert assumptions["cement_type"] == "OPC"
    assert isinstance(assumptions["clinker_factor"], (int, float))
    assert assumptions["clinker_factor"] == pytest.approx(0.95)
    for key in (
        "opc_assumption_note",
        "scm_strength_contribution_note",
        "gel_space_ratio_constants_source",
    ):
        assert isinstance(assumptions[key], str)
        assert len(assumptions[key]) > 20


def test_schema_fit_types(result_json):
    fit = result_json["fit"]
    for key in ("A", "n", "r2_log_space", "r2_mpa_space", "residual_std_mpa"):
        assert key in fit
        assert isinstance(fit[key], (int, float))
    assert isinstance(fit["n_sanity_band_reported"], list)
    assert len(fit["n_sanity_band_reported"]) == 2
    assert isinstance(fit["n_within_sanity_band"], bool)


def test_schema_rows(result_json):
    rows = result_json["rows"]
    assert isinstance(rows["total"], int)
    assert isinstance(rows["used"], int)
    assert isinstance(rows["excluded"], int)
    assert rows["total"] == rows["used"] + rows["excluded"]
    assert rows["total"] == 1030
    assert isinstance(rows["excluded_reasons"], dict)


def test_schema_bucket_residuals_structure(result_json):
    for bucket_group in ("age_bucket_residuals_mpa", "scm_fraction_bucket_residuals_mpa"):
        buckets = result_json[bucket_group]
        assert isinstance(buckets, dict)
        assert len(buckets) > 0
        for label, stats in buckets.items():
            assert "count" in stats
            assert isinstance(stats["count"], int)
            if stats["count"] > 0:
                assert isinstance(stats["residual_mean_mpa"], (int, float))
                assert isinstance(stats["residual_std_mpa"], (int, float))


def test_age_buckets_cover_all_used_rows(result_json):
    total_bucketed = sum(
        b["count"] for b in result_json["age_bucket_residuals_mpa"].values()
    )
    assert total_bucketed == result_json["rows"]["used"]


def test_scm_buckets_cover_all_used_rows(result_json):
    total_bucketed = sum(
        b["count"] for b in result_json["scm_fraction_bucket_residuals_mpa"].values()
    )
    assert total_bucketed == result_json["rows"]["used"]


# --------------------------------------------------------------------------
# n: finite and positive only. The [1.5, 4.5] sanity band is INFORMATION,
# never a pass bar (WP-F honesty constraint).
# --------------------------------------------------------------------------

def test_n_is_finite_and_positive(result_json):
    n = result_json["fit"]["n"]
    assert n == n, "n must not be NaN"  # noqa: PLR0124 (NaN != NaN check)
    assert n not in (float("inf"), float("-inf"))
    assert n > 0


def test_n_sanity_band_is_reported_not_gated(result_json):
    """The band is recorded as information; whether n falls inside it must not
    affect any other field or cause a failure by itself."""
    fit = result_json["fit"]
    assert fit["n_sanity_band_reported"] == [1.5, 4.5]
    # n_within_sanity_band is purely informational -- both outcomes are valid.
    assert fit["n_within_sanity_band"] in (True, False)


def test_r2_values_are_reported_whatever_they_are(result_json):
    """R^2 is a finding, not a target: only sanity-check it is a real number
    in a mathematically valid range for a fit that (usually) beats the mean."""
    fit = result_json["fit"]
    assert -10.0 <= fit["r2_log_space"] <= 1.0
    assert -10.0 <= fit["r2_mpa_space"] <= 1.0


def test_residual_std_is_nonnegative(result_json):
    assert result_json["fit"]["residual_std_mpa"] >= 0


# --------------------------------------------------------------------------
# The fence: zero runtime integration.
# --------------------------------------------------------------------------

def test_no_xgboost_import():
    """Importing the fit script module must not pull in xgboost (src/models.py's
    dependency) -- WP-F is a standalone evidence script, not a runtime path.

    Runs in a FRESH interpreter (the test_cli_never_imports_streamlit pattern):
    asserting on this process's sys.modules is order-dependent -- any earlier test
    that touches StrengthPredictor imports xgboost and poisons the check."""
    code = (
        "import importlib.util, sys; "
        f"spec = importlib.util.spec_from_file_location('fit_powers_module', {str(SCRIPT_PATH)!r}); "
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
        "assert 'xgboost' not in sys.modules, 'fit_powers.py must not import xgboost'; "
        "assert 'src.models' not in sys.modules, 'fit_powers.py must not import src.models'"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=str(REPO_ROOT))
    assert r.returncode == 0, r.stderr


def test_no_ui_import_in_script_source():
    source = SCRIPT_PATH.read_text()
    assert "from ui" not in source
    assert "import ui" not in source


def test_script_source_does_not_reference_predictor_modules():
    """WP-F must never touch src/models.py or the predictor paths."""
    source = SCRIPT_PATH.read_text()
    assert "src.models" not in source
    assert "src.optimizer" not in source


# --------------------------------------------------------------------------
# Honesty constraints on the fitted A/n themselves.
# --------------------------------------------------------------------------

def test_only_A_and_n_are_the_fitted_parameters(result_json):
    """Guard against scope creep: the fit block must contain exactly A and n
    as free parameters (plus the reported diagnostics), never additional
    tuned constants."""
    fit_keys = set(result_json["fit"].keys())
    expected = {
        "A", "n", "n_sanity_band_reported", "n_within_sanity_band",
        "r2_log_space", "r2_mpa_space", "residual_std_mpa",
    }
    assert fit_keys == expected


def test_gel_space_ratio_constants_are_powers_canonical(result_json):
    """0.68 / 0.32 are Powers' declared constants, not fitted -- pinned here so
    a future edit that starts "fitting" them trips this test."""
    formula = result_json["formulas"]["gel_space_ratio"]
    assert "0.68" in formula
    assert "0.32" in formula
