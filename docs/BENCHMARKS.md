# Backend benchmarks

Comparative numbers for the inverse-design backends, so "when to use which" (see [`docs/WORKFLOW.md`](WORKFLOW.md) §1) is backed by evidence rather than vibes.

**Metrics.** `mae_MPa` = mean |predicted − target| over the sampled cloud (lower = hits the target better). `cloud_spread` = mean per-parameter std of the cloud (higher = more design diversity). `in_support_pct` = share of the cloud inside the trusted data region (higher = more trustworthy predictions). `time_s` = wall-clock for the run.

Sampled clouds of **1500** mixes per (backend, target).

## Inverse-design backends

| backend | target_MPa | mae_MPa | cloud_spread | in_support_pct | time_s |
| --- | --- | --- | --- | --- | --- |
| flow | 25.0 | 8.62 | 67.04 | 6.5 | 0.18 |
| flow | 45.0 | 4.09 | 78.84 | 4.7 | 0.13 |
| flow | 65.0 | 4.93 | 71.68 | 2.9 | 0.13 |
| ga | 25.0 | 3.99 | 28.27 | 1.7 | 17.33 |
| ga | 45.0 | 1.08 | 41.51 | 0.0 | 66.16 |
| ga | 65.0 | 0.98 | 38.5 | 0.0 | 7.69 |
| aco | 25.0 | 0.63 | 7.05 | 0.0 | 1.94 |
| aco | 45.0 | 1.36 | 19.54 | 0.0 | 2.12 |
| aco | 65.0 | 0.7 | 19.94 | 0.7 | 1.96 |

## Age-conditioned flow (R7.2)

The flow now conditions on **(strength, age)**, so a fixed design age is served by the flow directly (before R7.2 a pinned age routed to the GA). Rows below pin age = 28 d — the tool's default; the age column is held exactly.

| backend | target_MPa | mae_MPa | cloud_spread | in_support_pct | time_s |
| --- | --- | --- | --- | --- | --- |
| flow | 25.0 | 7.1 | 58.96 | 23.7 | 0.15 |
| flow | 45.0 | 6.27 | 63.9 | 15.9 | 0.13 |
| flow | 65.0 | 3.4 | 53.32 | 10.1 | 0.13 |

## Multi-objective (NSGA-II)

NSGA maps a whole strength/carbon/cost front in one run, so it is reported separately (front-wide, not per-target).

| algorithm | front_size | mean_strength_MPa | in_support_pct | time_s |
| --- | --- | --- | --- | --- |
| NSGA-II | 60 | 62.6 | 5.0 | 0.33 |

## Reproduce

```bash
python -m scripts.benchmark_backends
```

Timings are machine-dependent (single-threaded CPU here) and the metaheuristics are stochastic, so treat the numbers as order-of-magnitude comparisons, not fixed values.
