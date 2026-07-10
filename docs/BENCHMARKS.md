# Backend benchmarks

Comparative numbers for the inverse-design backends, so "when to use which" (see [`docs/WORKFLOW.md`](WORKFLOW.md) §1) is backed by evidence rather than vibes.

**Metrics.** `mae_MPa` = mean |predicted − target| over the sampled cloud (lower = hits the target better). `cloud_spread` = mean per-parameter std of the cloud (higher = more design diversity). `in_support_pct` = share of the cloud inside the trusted data region (higher = more trustworthy predictions). `time_s` = wall-clock for the run.

Sampled clouds of **1500** mixes per (backend, target).

## Inverse-design backends

| backend | target_MPa | mae_MPa | cloud_spread | in_support_pct | time_s |
| --- | --- | --- | --- | --- | --- |
| flow | 25.0 | 8.84 | 66.72 | 6.5 | 0.18 |
| flow | 45.0 | 4.15 | 78.38 | 5.2 | 0.12 |
| flow | 65.0 | 4.92 | 72.34 | 3.3 | 0.12 |
| ga | 25.0 | 3.99 | 28.27 | 1.7 | 3.0 |
| ga | 45.0 | 1.08 | 41.51 | 0.0 | 3.14 |
| ga | 65.0 | 0.98 | 38.5 | 0.0 | 2.94 |
| aco | 25.0 | 0.63 | 7.05 | 0.0 | 1.52 |
| aco | 45.0 | 1.36 | 19.54 | 0.0 | 1.48 |
| aco | 65.0 | 0.7 | 19.94 | 0.7 | 1.53 |

## Age-conditioned flow (R7.2)

The flow now conditions on **(strength, age)**, so a fixed design age is served by the flow directly (before R7.2 a pinned age routed to the GA). Rows below pin age = 28 d — the tool's default; the age column is held exactly.

| backend | target_MPa | mae_MPa | cloud_spread | in_support_pct | time_s |
| --- | --- | --- | --- | --- | --- |
| flow | 25.0 | 7.01 | 60.11 | 23.1 | 0.12 |
| flow | 45.0 | 6.48 | 63.41 | 15.1 | 0.12 |
| flow | 65.0 | 3.43 | 54.48 | 9.7 | 0.12 |

## Robust mode (R7.3a)

**Robust mode is the tool's default** (see the Config tab), so this — not the non-robust table above — is the configuration users actually run. Under `robust=True` the GA/ACO objective switches from matching the mean prediction to matching the **conformal lower bound** while penalising out-of-support novelty, so `lower_bound_mae_MPa` (not median MAE) is the metric the optimizer was scored on. The flow does not recondition on `robust=True` inside `sample_posterior` — its robust handling is a downstream filter in `recommend_recipe`, not a resample — so no robust flow row is shown here; a row implying the flow's *sampling* changed would misrepresent what runs.

| backend | target_MPa | lower_bound_mae_MPa | cloud_spread | in_support_pct | time_s |
| --- | --- | --- | --- | --- | --- |
| ga | 25.0 | 2.23 | 27.77 | 89.3 | 3.23 |
| ga | 45.0 | 1.59 | 25.8 | 87.9 | 3.25 |
| ga | 65.0 | 2.44 | 16.81 | 63.6 | 3.2 |
| aco | 25.0 | 0.99 | 21.82 | 89.7 | 1.78 |
| aco | 45.0 | 1.7 | 15.82 | 100.0 | 1.64 |
| aco | 65.0 | 2.52 | 7.41 | 96.3 | 1.65 |

Compare `in_support_pct` against the non-robust table above: this is the trade robust mode buys — markedly higher in-support share, i.e. the in-support **guarantee** the roadmap's robust formulation targets, at some cost in target tightness / diversity.

## Multi-objective (NSGA-II)

NSGA maps a whole strength/carbon/cost front in one run, so it is reported separately (front-wide, not per-target). The robust row adds the in-support constraint (expect in-support ≈ 100%); `hypervolume` is computed for both rows against ONE shared reference point, so the two numbers are directly comparable — a smaller robust hypervolume is the expected, deliberate cost of restricting the front to the trusted data region.

| algorithm | front_size | mean_strength_MPa | in_support_pct | hypervolume | time_s |
| --- | --- | --- | --- | --- | --- |
| NSGA-II (non-robust) | 60 | 62.6 | 5.0 | 1082794.4 | 0.25 |
| NSGA-II (robust) | 45 | 42.9 | 100.0 | 820612.9 | 0.31 |

## Reproduce

```bash
python -m scripts.benchmark_backends
```

Timings are machine-dependent (single-threaded CPU here) and the metaheuristics are stochastic, so treat the numbers as order-of-magnitude comparisons, not fixed values.
