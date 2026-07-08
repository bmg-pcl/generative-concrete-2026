# Backend benchmarks

Comparative numbers for the inverse-design backends, so "when to use which" (see [`docs/WORKFLOW.md`](WORKFLOW.md) §1) is backed by evidence rather than vibes.

**Metrics.** `mae_MPa` = mean |predicted − target| over the sampled cloud (lower = hits the target better). `cloud_spread` = mean per-parameter std of the cloud (higher = more design diversity). `in_support_pct` = share of the cloud inside the trusted data region (higher = more trustworthy predictions). `time_s` = wall-clock for the run.

Sampled clouds of **1500** mixes per (backend, target).

## Inverse-design backends

| backend | target_MPa | mae_MPa | cloud_spread | in_support_pct | time_s |
| --- | --- | --- | --- | --- | --- |
| flow | 25.0 | 5.98 | 71.6 | 16.7 | 0.22 |
| flow | 45.0 | 3.99 | 77.73 | 5.1 | 0.15 |
| flow | 65.0 | 3.96 | 70.04 | 2.2 | 0.15 |
| ga | 25.0 | 3.99 | 28.27 | 1.7 | 3.85 |
| ga | 45.0 | 1.08 | 41.51 | 0.0 | 4.18 |
| ga | 65.0 | 0.98 | 38.5 | 0.0 | 3.86 |
| aco | 25.0 | 0.63 | 7.05 | 0.0 | 2.06 |
| aco | 45.0 | 1.36 | 19.54 | 0.0 | 2.01 |
| aco | 65.0 | 0.7 | 19.94 | 0.7 | 2.07 |

## Multi-objective (NSGA-II)

NSGA maps a whole strength/carbon/cost front in one run, so it is reported separately (front-wide, not per-target).

| algorithm | front_size | mean_strength_MPa | in_support_pct | time_s |
| --- | --- | --- | --- | --- |
| NSGA-II | 60 | 62.6 | 5.0 | 0.34 |

## Reproduce

```bash
python -m scripts.benchmark_backends
```

Timings are machine-dependent (single-threaded CPU here) and the metaheuristics are stochastic, so treat the numbers as order-of-magnitude comparisons, not fixed values.
