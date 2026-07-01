# R1 — Trustworthy Numbers

**Status:** Draft · **Effort:** ~1–2 days · **Depends on:** nothing · **Unblocks:** R2 (NSGA constraint plumbing), R3.1

## Motivation

Phase C added prediction intervals and an out-of-support (OOS) score, but left three
debts that undermine the numbers users see:

1. The "90%" intervals are **raw trained quantiles with unmeasured coverage** — XGBoost
   quantile regression typically under-covers on unseen data, and no test verifies the
   claim on held-out points.
2. The novelty threshold (`1.5` in `models.in_support` and hardcoded again in
   `ui_logic.compute_metrics`/`recommend_recipe`) is **arbitrary**, not calibrated.
3. The optimizers still **optimize the mean**: NSGA/GA rank a mix predicting 60 ± 25 MPa
   above one predicting 58 ± 5, and nothing steers the search away from sparse regions —
   the OOS guard warns *after* the fact.

## 1. Split-conformal calibration of the intervals

### Current state
`src/models.py`: `train()` fits a multi-quantile model (α = 0.05/0.5/0.95) on `X_train`
(80% split); `predict_interval()` returns the raw quantiles; `predict_variance()` is half
the interval width.

### Design
- Re-split inside `train()`: `X_train` → **fit** (75%) and **calibration** (25%),
  keeping the existing 20% test split untouched (it is reserved for the coverage *test*).
- Fit the quantile model on the fit fold. On the calibration fold compute conformity
  scores `s_i = max(q_lo(x_i) − y_i, y_i − q_hi(x_i))` and the conformal correction
  `q̂ = ⌈(n+1)·0.90⌉/n`-th empirical quantile of `{s_i}`.
- `predict_interval()` returns `[q_lo(x) − q̂, q_hi(x) + q̂]` (median unchanged). Persist
  `q̂` in `models/support.npz` (new key `conformal_q`); fall back to `q̂ = 0` with a
  logged warning when absent (old artifacts).
- Retrain and commit regenerated artifacts.

### Acceptance gates
- New test `tests/test_models.py::test_interval_coverage`: empirical coverage of the 90%
  interval on the held-out test set (n≈206) lies in **[0.84, 0.97]** (±3 binomial σ around
  0.90). Runs in CI.
- Existing interval tests (ordering, positive width, `variance == half-width`) still pass.
- `docs/TECHNICAL_REPORT.md` §5.2 gains the measured coverage number.

## 2. Data-calibrated novelty threshold

### Current state
`models.novelty()` returns kNN-distance / in-training reference; `in_support(threshold=1.5)`
uses a hand-picked constant, duplicated inline in `ui_logic`.

### Design
- At `train()` time, score the held-out test fold's novelty against the fit-fold support
  set and store the **95th percentile** as the threshold in `support.npz` (key
  `novelty_threshold`).
- `in_support()` defaults to the stored threshold; `1.5` remains only as a fallback for
  old artifacts. Remove the inline `<= 1.5` copies in `ui_logic` — call
  `predictor.in_support()` so there is one source of truth.

### Acceptance gates
- Test: ≥ 90% of held-out rows are `in_support` under the stored threshold (they are real
  mixes; the calibration targets 95%, the gate allows sampling slack).
- Test: the all-envelope-corners probe remains out of support.
- Grep gate: no hardcoded `1.5` novelty comparisons outside `models.py`.

## 3. Robust optimization (use the uncertainty, don't just display it)

### Design
- **Scalarized GA/SA** (`ui_logic.scalarized_fitness`): add `robust: bool = False`; when
  True the strength term uses the conformal **lower bound** `lo(x)` instead of the mean.
- **Inverse designers** (`generative_ga._InverseDesignerBase._make_objective`): add an OOS
  penalty `error += w_oos · max(0, novelty(x) − threshold)` with `w_oos ≈ 10` (MPa-scale
  units; tune so a strongly-OOS mix cannot win). Batch novelty per generation to bound
  the kNN cost.
- **NSGA** (`src/nsga.py::MixDesignProblem`): set `n_ieq_constr=1` with
  `G = novelty(X) − threshold` so the returned front is **in-support by construction**;
  optionally expose `robust=True` to optimize `−lo(x)` instead of `−mean`.
- **UI:** a "Robust mode (optimize the guaranteed-strength bound, stay in-data)" toggle in
  the Config tab, default **ON** for the recommended recipe and NSGA, threaded like
  `carbon_kwargs`. The recipe caption states which bound was optimized.

### Acceptance gates
- Test: robust NSGA front is 100% `in_support`.
- Test: robust `recommend_recipe(target)` returns a mix with `interval_lo ≥ target − 5`
  and `in_support == True` for target = 40 MPa.
- Test: with robust off, behavior is unchanged (existing tests green).

## Out of scope
- Conditional/Mondrian conformal (per-region coverage) — follow-up if global coverage
  masks local failure.
- Replacing kNN novelty with a density model.

## Risks
- Conformal widening will make intervals honest and therefore **wider**; the UI copy must
  frame this as increased honesty, not decreased quality.
- The OOS constraint can shrink NSGA's feasible region; if fronts collapse, relax the
  threshold percentile (95 → 97.5) rather than dropping the constraint.

## Implementation notes (sonnet-ready)

- **Method is CQR** (Conformalized Quantile Regression — Romano, Patterson & Candès,
  NeurIPS 2019). Do not hand-roll a different conformal variant. Symmetric conformity
  score `s_i = max(q_lo(x_i) − y_i, y_i − q_hi(x_i))`; correction `q̂` = the k-th smallest
  `s` with `k = ceil((n_cal + 1) · 0.90)`; the interval is `[q_lo − q̂, q_hi + q̂]`. The
  quantile indexing is the easy thing to get subtly wrong — **the coverage test is the
  safety net; if it fails, fix the index, do NOT widen the test's [0.84, 0.97] band.**
- **The mean model is unchanged.** Keep `self.model` trained on the full 80% `X_train`
  (RMSE/R² must stay 4.65/0.92). Only the *quantile* model uses a fit/cal sub-split of
  `X_train`. The 20% test split is read only by `test_interval_coverage` — never train on it.
- **Robust defaults:** keep every function signature at `robust=False` so existing tests
  stay green; only the *UI* passes `robust=True`. Do not change function defaults.
- **Don't "tune" `w_oos` by feel** — set it to the smallest value that makes the robust
  tests pass; start at 10 and raise only if a strongly-OOS mix still wins the front.
- Sequencing within R1: do R1.1 then R1.2 (both add keys to `support.npz`; write both in
  one `train()` pass and regenerate artifacts once), then R1.3.
