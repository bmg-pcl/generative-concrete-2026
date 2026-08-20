"""
fit_powers.py -- WP-F phase 1: Powers gel-space-ratio calibration (evidence only).

See docs/specs/R8.0-carbon-chemistry-deepening.md, WP-F. This script is a
STANDALONE FITTING EXERCISE with ZERO runtime integration: it reads the UCI
concrete dataset, pushes each row's cement chemistry through the *existing,
UNCALIBRATED* `hydration_kinetics` model (src/chemistry_advanced.py) to get a
degree of hydration alpha, converts alpha to Powers' classical gel-space ratio
X, and fits exactly TWO numbers -- Powers' A and n in `f = A * X^n` -- against
the dataset's 1030 measured compressive strengths. It writes
`models/powers_calibration.json` and prints a markdown report. Nothing in
`src/models.py`, the predictors, or any runtime path reads this artifact (yet)
-- integration is a future spec that will cite these numbers.

THE FENCE (read before touching this file):
  - Exactly two fitted parameters: A and n. The hydration model's own kinetic
    constants -- the 0.15 rate coefficient and 0.38 water-availability divisor
    inside `hydration_kinetics`, the C3S/C2S/C3A/C4AF heat coefficients, the
    pozzolanic/latent-hydraulic reactivity tables -- are NOT refit, NOT
    tweaked, and NOT "sensitivity-analyzed into new defaults" by this script.
    The claim under test is precisely: "alpha(t) as currently modelled, pushed
    through Powers, correlates with measured strength THIS WELL" -- not "what
    combination of knobs produces the best-looking R^2".
  - R^2 is a FINDING, not a target. Whatever the arithmetic yields is what gets
    reported and written to the JSON. Do not iterate on the hydration layer,
    the oxide table, or the OPC assumption below to chase a higher number.
  - This fit ignores slag/ash contributions to strength: alpha comes from
    `hydration_kinetics`, which models CLINKER hydration only (see its
    docstring). SCM (slag/fly-ash) pozzolanic strength gain is not represented
    in X at all. This is the EXPECTED DOMINANT residual source on high-SCM
    mixes, and the per-SCM-fraction residual breakdown below is included
    specifically to let that show up in the numbers rather than being
    corrected for.

OPC ASSUMPTION (documented, not hidden):
    The UCI dataset does not record a cement type per row -- it is one
    unlabelled "cement" mass column. This script assumes every row's cement is
    Ordinary Portland Cement (the "OPC" record in data/oxide_compositions.json,
    the same registry path `analyze_mix` uses) with a fixed clinker_factor of
    0.95. Both the OPC oxide composition and the clinker_factor are therefore
    IDENTICAL for all 1030 rows -- only water/cement ratio, cement dosage, and
    curing age vary row to row. If any row's cement is actually a blend (e.g.
    PLC, LC3), this assumption is wrong for that row and its residual is
    attributed to unmodelled hydration, not to Powers' fit itself.

POWERS' GEL-SPACE RATIO (Powers & Brownyard, 1948; Powers, 1958):
    X = 0.68 * alpha / (0.32 * alpha + w/c)
    The classical "Powers' law" strength relation is f = A * X^n with n
    canonically ~3 for OPC pastes; A and n are fit here by ordinary least
    squares on the log-linearized form:
        ln(f) = ln(A) + n * ln(X)

Determinism: fixed row ordering (as loaded from the UCI file, no shuffling),
no randomness, no timestamps or paths embedded in the JSON -- two runs produce
byte-identical output. The orchestrator stamps commit provenance separately.

Usage:
    python scripts/fit_powers.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

# Allow `python scripts/fit_powers.py` (repo root not on sys.path by default)
# as well as `python -m scripts.fit_powers` / import from tests.
_REPO_ROOT_FOR_IMPORT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT_FOR_IMPORT not in sys.path:
    sys.path.insert(0, _REPO_ROOT_FOR_IMPORT)

from src.chemistry_advanced import (  # noqa: E402
    bogue_calculation,
    clinker_factor_for,
    hydration_kinetics,
    load_oxide_compositions,
)
from src.data_fetcher import load_data  # noqa: E402

# ----------------------------------------------------------------------------
# Constants declared up front, per the honesty fence -- these are Powers'
# canonical constants and the OPC assumption, not fitted quantities.
# ----------------------------------------------------------------------------
CEMENT_TYPE = "OPC"
CLINKER_FACTOR = 0.95  # OPC assumption, matches data/oxide_compositions.json

GEL_SPACE_RATIO_CONSTANT_A = 0.68  # Powers & Brownyard (1948)
GEL_SPACE_RATIO_CONSTANT_B = 0.32  # Powers & Brownyard (1948)
GEL_SPACE_RATIO_FORMULA = (
    f"X = {GEL_SPACE_RATIO_CONSTANT_A}*alpha / "
    f"({GEL_SPACE_RATIO_CONSTANT_B}*alpha + w_c)"
)
POWERS_LAW_FORMULA = "f = A * X^n"
LOG_LINEAR_FORMULA = "ln(f) = ln(A) + n*ln(X)"

# Powers' canonical n is ~3; this band is REPORTED as information only (WP-F
# gate: asserted finite and positive, never gated on falling inside the band).
N_SANITY_BAND = (1.5, 4.5)

# Age buckets (days). Boundaries are chosen to land on the real gaps in the
# UCI dataset's discrete age values ({1,3,7} | {14,28} | {56,90} | {91,...}),
# so every row falls unambiguously into exactly one bucket.
AGE_BUCKETS = [
    ("<=7", lambda age: age <= 7),
    ("14-28", lambda age: 14 <= age <= 28),
    ("56-90", lambda age: 56 <= age <= 90),
    (">=91", lambda age: age >= 91),
]

# SCM (slag + fly ash) mass fraction of total cementitious material
# (cement + slag + ash). Buckets chosen to separate "no SCM" mixes (where this
# fit's clinker-only alpha should be most honest) from increasingly
# SCM-heavy mixes (where the missing pozzolanic-strength term should show up
# as a growing residual).
SCM_FRACTION_BUCKETS = [
    ("0%", lambda f: f <= 0.0),
    ("0-25%", lambda f: 0.0 < f <= 0.25),
    ("25-50%", lambda f: 0.25 < f <= 0.50),
    ("50%+", lambda f: f > 0.50),
]

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
OUTPUT_JSON_PATH = os.path.join(_REPO_ROOT, "models", "powers_calibration.json")


def compute_alpha_and_x(df) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (w_c, alpha, X) per row using the existing, uncalibrated
    hydration_kinetics model via the same registry path analyze_mix uses.

    The OPC oxide record (and therefore the Bogue clinker phases) is identical
    for every row -- only w/c, cement dosage, and age vary -- so `phases` is
    computed once outside the loop.
    """
    oxide_compositions = load_oxide_compositions()
    cement_record = oxide_compositions[CEMENT_TYPE]
    phases = bogue_calculation(cement_record)
    clinker_factor = clinker_factor_for(CEMENT_TYPE, oxide_compositions)
    assert clinker_factor == CLINKER_FACTOR, (
        f"registry clinker_factor for {CEMENT_TYPE} ({clinker_factor}) drifted "
        f"from the documented WP-F assumption ({CLINKER_FACTOR})"
    )

    n = len(df)
    w_c = np.empty(n, dtype=float)
    alpha = np.empty(n, dtype=float)

    cement = df["cement"].to_numpy(dtype=float)
    water = df["water"].to_numpy(dtype=float)
    age = df["age"].to_numpy(dtype=float)

    for i in range(n):
        w_c[i] = water[i] / cement[i]
        state = hydration_kinetics(
            phases,
            w_c_ratio=w_c[i],
            age_days=age[i],
            cement_mass_kg_m3=cement[i],
            clinker_factor=clinker_factor,
        )
        alpha[i] = state.degree_of_hydration

    X = (GEL_SPACE_RATIO_CONSTANT_A * alpha) / (GEL_SPACE_RATIO_CONSTANT_B * alpha + w_c)
    return w_c, alpha, X


def fit_powers_law(X: np.ndarray, f: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Log-linear least squares: ln(f) = ln(A) + n*ln(X). Returns (A, n, design)."""
    ln_X = np.log(X)
    ln_f = np.log(f)
    design = np.column_stack([np.ones_like(ln_X), ln_X])
    coeffs, _residuals, _rank, _sv = np.linalg.lstsq(design, ln_f, rcond=None)
    ln_A, n = coeffs
    A = float(np.exp(ln_A))
    return A, float(n), design


def r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    ss_res = float(np.sum((observed - predicted) ** 2))
    ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def bucket_residual_stats(
    residuals: np.ndarray, keys: np.ndarray, buckets: list[tuple[str, object]]
) -> dict[str, dict]:
    """Mean/std/count of residuals (MPa) within each named bucket, in bucket
    declaration order (ordering is fixed -> deterministic JSON)."""
    out: dict[str, dict] = {}
    for label, predicate in buckets:
        mask = np.array([bool(predicate(k)) for k in keys])
        count = int(mask.sum())
        if count == 0:
            out[label] = {"count": 0, "residual_mean_mpa": None, "residual_std_mpa": None}
            continue
        bucket_residuals = residuals[mask]
        out[label] = {
            "count": count,
            "residual_mean_mpa": round(float(np.mean(bucket_residuals)), 4),
            "residual_std_mpa": round(float(np.std(bucket_residuals, ddof=1)), 4)
            if count > 1
            else 0.0,
        }
    return out


def run() -> dict:
    df = load_data()
    rows_total = len(df)

    w_c, alpha, X = compute_alpha_and_x(df)
    f = df["strength"].to_numpy(dtype=float)
    age = df["age"].to_numpy(dtype=float)
    cement = df["cement"].to_numpy(dtype=float)
    slag = df["slag"].to_numpy(dtype=float)
    ash = df["ash"].to_numpy(dtype=float)
    scm_fraction = (slag + ash) / (cement + slag + ash)

    # Guard: X > 0 and f > 0 only. Report exclusions explicitly (WP-F gate).
    valid_X = np.isfinite(X) & (X > 0)
    valid_f = np.isfinite(f) & (f > 0)
    keep = valid_X & valid_f
    n_excluded_X = int((~valid_X).sum())
    n_excluded_f = int((~valid_f & valid_X).sum())
    rows_used = int(keep.sum())
    rows_excluded = rows_total - rows_used

    X_fit = X[keep]
    f_fit = f[keep]
    age_fit = age[keep]
    scm_fraction_fit = scm_fraction[keep]

    A, n, _design = fit_powers_law(X_fit, f_fit)

    ln_X_fit = np.log(X_fit)
    ln_f_fit = np.log(f_fit)
    ln_f_pred = np.log(A) + n * ln_X_fit
    r2_log = r_squared(ln_f_fit, ln_f_pred)

    f_pred = A * X_fit ** n
    r2_mpa = r_squared(f_fit, f_pred)

    residuals_mpa = f_fit - f_pred
    residual_std_mpa = float(np.std(residuals_mpa, ddof=1))

    age_bucket_residuals = bucket_residual_stats(residuals_mpa, age_fit, AGE_BUCKETS)
    scm_bucket_residuals = bucket_residual_stats(
        residuals_mpa, scm_fraction_fit, SCM_FRACTION_BUCKETS
    )

    assert np.isfinite(n) and n > 0, f"fitted n must be finite and positive, got {n}"
    n_within_sanity_band = bool(N_SANITY_BAND[0] <= n <= N_SANITY_BAND[1])

    result = {
        "description": (
            "WP-F phase 1: Powers gel-space-ratio calibration. Evidence artifact "
            "only -- not read by any runtime path. See docs/specs/"
            "R8.0-carbon-chemistry-deepening.md, WP-F, and this script's "
            "module docstring for the full honesty fence."
        ),
        "formulas": {
            "gel_space_ratio": GEL_SPACE_RATIO_FORMULA,
            "powers_law": POWERS_LAW_FORMULA,
            "log_linear_fit": LOG_LINEAR_FORMULA,
        },
        "assumptions": {
            "cement_type": CEMENT_TYPE,
            "clinker_factor": CLINKER_FACTOR,
            "opc_assumption_note": (
                "The UCI dataset has one unlabelled 'cement' mass column with no "
                "cement-type field. Every row is assumed to be OPC (data/"
                "oxide_compositions.json 'OPC' record) with clinker_factor=0.95. "
                "This is identical for all rows -- only w/c, cement dosage, and "
                "age vary row to row. Rows that are actually blended cement "
                "(PLC, LC3, etc.) violate this assumption and their residual is "
                "attributed to unmodelled hydration, not to the Powers fit."
            ),
            "scm_strength_contribution_note": (
                "alpha comes from hydration_kinetics, which models CLINKER "
                "hydration only (see its docstring in src/chemistry_advanced.py). "
                "Slag and fly-ash pozzolanic/latent-hydraulic strength "
                "contributions are NOT represented in X. This is the expected "
                "dominant residual source on high-SCM mixes -- see "
                "scm_fraction_bucket_residuals_mpa below."
            ),
            "gel_space_ratio_constants_source": "Powers & Brownyard (1948); Powers (1958)",
        },
        "fit": {
            "A": round(A, 6),
            "n": round(n, 6),
            "n_sanity_band_reported": list(N_SANITY_BAND),
            "n_within_sanity_band": n_within_sanity_band,
            "r2_log_space": round(r2_log, 6),
            "r2_mpa_space": round(r2_mpa, 6),
            "residual_std_mpa": round(residual_std_mpa, 6),
        },
        "rows": {
            "total": rows_total,
            "used": rows_used,
            "excluded": rows_excluded,
            "excluded_reasons": {
                "X_not_finite_or_not_positive": n_excluded_X,
                "strength_not_finite_or_not_positive_given_X_valid": n_excluded_f,
            },
        },
        "age_bucket_residuals_mpa": age_bucket_residuals,
        "scm_fraction_bucket_residuals_mpa": scm_bucket_residuals,
    }
    return result


def format_report(result: dict) -> str:
    fit = result["fit"]
    rows = result["rows"]
    lines = []
    lines.append("# WP-F Phase 1 -- Powers Calibration Report")
    lines.append("")
    lines.append(
        "Fits Powers' classical gel-space-ratio strength law "
        f"(`{POWERS_LAW_FORMULA}`) against the UIUC/UCI concrete dataset, using "
        "alpha (degree of hydration) from the EXISTING, UNCALIBRATED "
        "`hydration_kinetics` model. Only A and n are fitted; the hydration "
        "model's own kinetic constants are untouched."
    )
    lines.append("")
    lines.append("## Formulas")
    lines.append(f"- Gel-space ratio: `{GEL_SPACE_RATIO_FORMULA}`")
    lines.append(f"- Powers' law: `{POWERS_LAW_FORMULA}`")
    lines.append(f"- Log-linear fit: `{LOG_LINEAR_FORMULA}`")
    lines.append("")
    lines.append("## OPC assumption")
    lines.append(result["assumptions"]["opc_assumption_note"])
    lines.append("")
    lines.append("## Fitted parameters")
    lines.append(f"- A = {fit['A']}")
    lines.append(f"- n = {fit['n']}  (Powers' canonical n ~ 3; sanity band "
                  f"{fit['n_sanity_band_reported']} reported, not gated; "
                  f"within band: {fit['n_within_sanity_band']})")
    lines.append(f"- R^2 (log space, the space actually fit): {fit['r2_log_space']}")
    lines.append(f"- R^2 (MPa space, f vs A*X^n): {fit['r2_mpa_space']}")
    lines.append(f"- Residual std (MPa, sample ddof=1): {fit['residual_std_mpa']}")
    lines.append("")
    lines.append(
        f"## Rows: {rows['used']} used / {rows['total']} total "
        f"({rows['excluded']} excluded)"
    )
    for reason, count in rows["excluded_reasons"].items():
        lines.append(f"- {reason}: {count}")
    lines.append("")
    lines.append("## Residuals (MPa, observed minus predicted) by age bucket")
    for label, stats in result["age_bucket_residuals_mpa"].items():
        lines.append(
            f"- age {label} days: n={stats['count']}, "
            f"mean={stats['residual_mean_mpa']}, std={stats['residual_std_mpa']}"
        )
    lines.append("")
    lines.append(
        "## Residuals (MPa) by SCM mass fraction (slag+ash / total cementitious)"
    )
    for label, stats in result["scm_fraction_bucket_residuals_mpa"].items():
        lines.append(
            f"- SCM {label}: n={stats['count']}, "
            f"mean={stats['residual_mean_mpa']}, std={stats['residual_std_mpa']}"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append(
        "This is a FINDING, not a target: the R^2 above is exactly what the "
        "placeholder hydration kinetics, pushed through Powers' law, explain of "
        "measured strength -- nothing was tuned upstream to raise it. Because "
        "alpha models clinker hydration only, the fit ignores slag/ash "
        "contributions to strength entirely; the SCM-fraction residual "
        "breakdown above is expected to show growing/systematic residuals as "
        "SCM fraction increases, and the age-bucket breakdown is expected to "
        "show whatever the placeholder time-kinetics get wrong at early vs. "
        "late ages. Neither breakdown was used to adjust the fit."
    )
    return "\n".join(lines)


def main() -> None:
    result = run()
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    print(format_report(result))
    print("")
    print(f"Wrote {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()
