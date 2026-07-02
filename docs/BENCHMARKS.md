# Backend benchmarks

Comparative numbers for the inverse-design backends, so "when to use which" (see [`docs/WORKFLOW.md`](WORKFLOW.md) §1) is backed by evidence rather than vibes.

**Metrics.** `mae_MPa` = mean |predicted − target| over the sampled cloud (lower = hits the target better). `cloud_spread` = mean per-parameter std of the cloud (higher = more design diversity). `in_support_pct` = share of the cloud inside the trusted data region (higher = more trustworthy predictions). `time_s` = wall-clock for the run.

Sampled clouds of **1500** mixes per (backend, target).

## Inverse-design backends

| backend | target_MPa | mae_MPa | cloud_spread | in_support_pct | time_s |
| --- | --- | --- | --- | --- | --- |
| flow | 25.0 | 7.23 | 70.91 | 19.9 | 0.18 |
| flow | 45.0 | 4.82 | 78.93 | 7.3 | 0.14 |
| flow | 65.0 | 4.19 | 72.24 | 2.7 | 0.15 |
| ga | 25.0 | 2.75 | 35.36 | 13.3 | 1.41 |
| ga | 45.0 | 1.09 | 27.22 | 1.3 | 1.44 |
| ga | 65.0 | 1.8 | 28.33 | 0.0 | 1.44 |
| aco | 25.0 | 7.69 | 5.17 | 0.0 | 0.85 |
| aco | 45.0 | 1.11 | 36.61 | 0.0 | 0.83 |
| aco | 65.0 | 1.63 | 57.07 | 0.0 | 0.75 |

## Multi-objective (NSGA-II)

NSGA maps a whole strength/carbon/cost front in one run, so it is reported separately (front-wide, not per-target).

| algorithm | front_size | mean_strength_MPa | in_support_pct | time_s |
| --- | --- | --- | --- | --- |
| NSGA-II | 60 | 66.1 | 0.0 | 0.23 |

## Reproduce

```bash
python -m scripts.benchmark_backends
```

Timings are machine-dependent (single-threaded CPU here) and the metaheuristics are stochastic, so treat the numbers as order-of-magnitude comparisons, not fixed values.
