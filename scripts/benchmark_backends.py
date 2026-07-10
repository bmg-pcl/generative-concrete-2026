"""Benchmark the inverse-design backends (flow / GA / ACO) and NSGA runs.

For a small grid of target strengths, each metaheuristic/flow backend samples a mix
cloud; we measure how well it hits the target, how spread out it is, how much of it
stays inside the trusted data region, and how long it took. NSGA is reported
separately (it maps a whole front rather than one target).

R7.3a adds a ROBUST section: robust mode is the tool's default, so it is the mode
that should be evidenced, not just the non-robust baseline above. GA/ACO under
robust=True optimize the conformal LOWER bound (not the mean), so their target-match
metric is lower-bound MAE, not median MAE -- reporting median MAE under "robust"
would misrepresent what the optimizer was actually scored on. The flow does NOT
recondition on robust inside sample_posterior (robust selection for the flow is a
downstream filter in recommend_recipe, not a resample), so no "robust flow" row is
emitted here -- one would misleadingly imply the flow's sampling changed.

Writes a markdown table to docs/BENCHMARKS.md. Not part of CI (timing-sensitive); a
fast smoke test (tests/test_benchmark.py) exercises the plumbing with a tiny budget.

    python -m scripts.benchmark_backends          # full run → docs/BENCHMARKS.md
    python -m scripts.benchmark_backends --quick   # tiny budget, print only
"""
import argparse
import os
import time

import numpy as np

from src.bayesian import BayesFlowExplorer

TARGETS = [25.0, 45.0, 65.0]
DEFAULT_BACKENDS = ["flow", "ga", "aco"]
ROBUST_BACKENDS = ["ga", "aco"]  # the flow is excluded from the robust section — see module docstring
BENCH_COLUMNS = ["backend", "target_MPa", "mae_MPa", "cloud_spread", "in_support_pct", "time_s"]
ROBUST_BENCH_COLUMNS = ["backend", "target_MPa", "lower_bound_mae_MPa", "cloud_spread", "in_support_pct", "time_s"]
NSGA_COLUMNS = ["algorithm", "front_size", "mean_strength_MPa", "in_support_pct", "time_s"]
ROBUST_NSGA_COLUMNS = ["algorithm", "front_size", "mean_strength_MPa", "in_support_pct", "hypervolume", "time_s"]
DOC_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "BENCHMARKS.md")


def _cloud_metrics(explorer, samples, target):
    predictor = explorer.predictor
    strength = predictor.predict_batch(samples)
    return {
        "mae_MPa": round(float(np.mean(np.abs(strength - target))), 2),
        "cloud_spread": round(float(np.mean(np.std(samples, axis=0))), 2),
        "in_support_pct": round(float(np.mean(predictor.in_support(samples)) * 100.0), 1),
    }


def _robust_cloud_metrics(explorer, samples, target):
    """Like _cloud_metrics, but scored on the conformal LOWER bound (what a robust
    search actually optimizes), not the median prediction."""
    predictor = explorer.predictor
    lo, _, _ = predictor.predict_interval(samples)
    return {
        "lower_bound_mae_MPa": round(float(np.mean(np.abs(lo - target))), 2),
        "cloud_spread": round(float(np.mean(np.std(samples, axis=0))), 2),
        "in_support_pct": round(float(np.mean(predictor.in_support(samples)) * 100.0), 1),
    }


def run_backend_benchmarks(explorer, backends, targets, n_samples=1500, seed=0, age=None):
    """Return a list of per-(backend, target) metric rows.

    Backends not currently available (e.g. 'flow' with no trained weights) are
    skipped; the returned rows say which ran, so a caller can note the skip. `age`
    pins the design age (R7.2: the flow now honors it instead of routing to the GA)."""
    rows = []
    for method in backends:
        if method == "flow" and explorer.amortized is None:
            continue  # no trained flow weights — skip rather than error
        for t in targets:
            np.random.seed(seed)  # best-effort determinism for the metaheuristics
            start = time.time()
            samples = explorer.sample_posterior(float(t), n_samples=n_samples,
                                                method=method, age=age)
            elapsed = time.time() - start
            row = {"backend": method, "target_MPa": float(t), "time_s": round(elapsed, 2)}
            row.update(_cloud_metrics(explorer, samples, t))
            rows.append(row)
    return rows


def run_robust_backend_benchmarks(explorer, backends, targets, n_samples=1500, seed=0):
    """R7.3a: robust=True rows for the metaheuristics ONLY (see module docstring for
    why the flow is excluded). Reports lower-bound MAE, the metric robust mode is
    actually scored on, plus in-support % (expected to be far higher than the
    non-robust table's, since the objective now penalises out-of-support novelty)."""
    rows = []
    for method in backends:
        for t in targets:
            np.random.seed(seed)
            start = time.time()
            samples = explorer.sample_posterior(float(t), n_samples=n_samples,
                                                method=method, robust=True)
            elapsed = time.time() - start
            row = {"backend": method, "target_MPa": float(t), "time_s": round(elapsed, 2)}
            row.update(_robust_cloud_metrics(explorer, samples, t))
            rows.append(row)
    return rows


def _hypervolume(strength, carbon, cost, ref_point):
    """Hypervolume of a strength/carbon/cost front against a shared reference point.

    Objectives are converted to minimisation (negate strength, since pymoo's HV
    indicator assumes minimisation) so both a robust and non-robust front can be
    scored against the SAME reference point for a fair size comparison."""
    from pymoo.indicators.hv import HV
    objs = np.column_stack([-np.asarray(strength), carbon, cost])
    return float(HV(ref_point=ref_point)(objs))


def run_nsga_benchmark(explorer, pop_size=60, n_gen=30, seed=1, robust=False):
    """Return a single NSGA metrics row (raw strength/carbon/cost arrays included,
    for hypervolume comparison across robust/non-robust runs), or None if pymoo is
    unavailable. robust=True adds the in-support constraint (expect in-support ~100%
    and, typically, a smaller/tighter front -- that trade is the point of this row)."""
    try:
        from src.nsga import run_nsga
    except Exception:
        return None
    predictor = explorer.predictor
    start = time.time()
    try:
        out = run_nsga(predictor, algorithm="nsga2", pop_size=pop_size, n_gen=n_gen,
                       random_seed=seed, robust=robust)
    except ImportError:
        return None
    elapsed = time.time() - start
    return {
        "algorithm": out["algorithm"],
        "front_size": int(out["front_size"]),
        "mean_strength_MPa": round(float(np.mean(out["strength"])), 1),
        "in_support_pct": round(float(np.mean(predictor.in_support(out["mixes"])) * 100.0), 1),
        "time_s": round(elapsed, 2),
        "_strength": out["strength"], "_carbon": out["carbon"], "_cost": out["cost"],
    }


def add_shared_hypervolume(row_a, row_b, margin=1.05):
    """Compute hypervolume for two NSGA rows (e.g. non-robust vs robust) against ONE
    shared reference point, so the two numbers are directly comparable. The reference
    point is the elementwise worst (in minimisation form) across both fronts, scaled
    outward by `margin` so every point strictly dominates it. Mutates both rows in
    place, adding a rounded 'hypervolume' key; no-op if either row is falsy."""
    if not row_a or not row_b:
        return
    obj_a = np.column_stack([-np.asarray(row_a["_strength"]), row_a["_carbon"], row_a["_cost"]])
    obj_b = np.column_stack([-np.asarray(row_b["_strength"]), row_b["_carbon"], row_b["_cost"]])
    ref = np.maximum(obj_a.max(axis=0), obj_b.max(axis=0)) * margin
    row_a["hypervolume"] = round(_hypervolume(row_a["_strength"], row_a["_carbon"], row_a["_cost"], ref), 1)
    row_b["hypervolume"] = round(_hypervolume(row_b["_strength"], row_b["_carbon"], row_b["_cost"], ref), 1)


def _md_table(columns, rows):
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(r[c]) for c in columns) + " |" for r in rows]
    return "\n".join([header, sep, *body])


def render_markdown(rows, nsga_row, n_samples, flow_available, age_rows=None,
                    robust_rows=None, nsga_robust_row=None):
    lines = [
        "# Backend benchmarks",
        "",
        "Comparative numbers for the inverse-design backends, so \"when to use which\" "
        "(see [`docs/WORKFLOW.md`](WORKFLOW.md) §1) is backed by evidence rather than vibes.",
        "",
        "**Metrics.** `mae_MPa` = mean |predicted − target| over the sampled cloud (lower = "
        "hits the target better). `cloud_spread` = mean per-parameter std of the cloud (higher "
        "= more design diversity). `in_support_pct` = share of the cloud inside the trusted "
        "data region (higher = more trustworthy predictions). `time_s` = wall-clock for the run.",
        "",
        f"Sampled clouds of **{n_samples}** mixes per (backend, target).",
    ]
    if not flow_available:
        lines += ["", "> The amortized **flow** backend is not benchmarked here: no trained "
                  "weights were found. Train it with `python -m src.amortized`, then re-run."]
    lines += ["", "## Inverse-design backends", "", _md_table(BENCH_COLUMNS, rows)]
    if age_rows:
        lines += ["", "## Age-conditioned flow (R7.2)", "",
                  "The flow now conditions on **(strength, age)**, so a fixed design age is served "
                  "by the flow directly (before R7.2 a pinned age routed to the GA). Rows below "
                  "pin age = 28 d — the tool's default; the age column is held exactly.", "",
                  _md_table(BENCH_COLUMNS, age_rows)]
    if robust_rows:
        lines += ["", "## Robust mode (R7.3a)", "",
                  "**Robust mode is the tool's default** (see the Config tab), so this — not the "
                  "non-robust table above — is the configuration users actually run. Under "
                  "`robust=True` the GA/ACO objective switches from matching the mean prediction "
                  "to matching the **conformal lower bound** while penalising out-of-support "
                  "novelty, so `lower_bound_mae_MPa` (not median MAE) is the metric the optimizer "
                  "was scored on. The flow does not recondition on `robust=True` inside "
                  "`sample_posterior` — its robust handling is a downstream filter in "
                  "`recommend_recipe`, not a resample — so no robust flow row is shown here; a "
                  "row implying the flow's *sampling* changed would misrepresent what runs.", "",
                  _md_table(ROBUST_BENCH_COLUMNS, robust_rows), "",
                  "Compare `in_support_pct` against the non-robust table above: this is the "
                  "trade robust mode buys — markedly higher in-support share, i.e. the "
                  "in-support **guarantee** the roadmap's robust formulation targets, at some "
                  "cost in target tightness / diversity."]
    if nsga_row is not None and nsga_robust_row is not None:
        lines += ["", "## Multi-objective (NSGA-II)", "",
                  "NSGA maps a whole strength/carbon/cost front in one run, so it is reported "
                  "separately (front-wide, not per-target). The robust row adds the in-support "
                  "constraint (expect in-support ≈ 100%); `hypervolume` is computed for both rows "
                  "against ONE shared reference point, so the two numbers are directly "
                  "comparable — a smaller robust hypervolume is the expected, deliberate cost "
                  "of restricting the front to the trusted data region.", "",
                  _md_table(ROBUST_NSGA_COLUMNS, [
                      {**nsga_row, "algorithm": nsga_row["algorithm"] + " (non-robust)"},
                      {**nsga_robust_row, "algorithm": nsga_robust_row["algorithm"] + " (robust)"},
                  ])]
    elif nsga_row is not None:
        lines += ["", "## Multi-objective (NSGA-II)", "",
                  "NSGA maps a whole strength/carbon/cost front in one run, so it is reported "
                  "separately (front-wide, not per-target).", "",
                  _md_table(NSGA_COLUMNS, [nsga_row])]
    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "python -m scripts.benchmark_backends",
        "```",
        "",
        "Timings are machine-dependent (single-threaded CPU here) and the metaheuristics are "
        "stochastic, so treat the numbers as order-of-magnitude comparisons, not fixed values.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Benchmark inverse-design backends.")
    ap.add_argument("--quick", action="store_true",
                    help="Tiny budget; print the table instead of writing docs/BENCHMARKS.md.")
    args = ap.parse_args(argv)

    explorer = BayesFlowExplorer()
    n_samples = 100 if args.quick else 1500
    targets = [45.0] if args.quick else TARGETS
    pop_size = 20 if args.quick else 60
    n_gen = 8 if args.quick else 30

    rows = run_backend_benchmarks(explorer, DEFAULT_BACKENDS, targets, n_samples=n_samples)
    # R7.2: age-conditioned flow at the default design age (28 d), if a flow is trained.
    age_rows = None
    if explorer.amortized is not None:
        age_rows = run_backend_benchmarks(explorer, ["flow"], targets,
                                          n_samples=n_samples, age=28.0)
    # R7.3a: robust mode is the tool's DEFAULT, so evidence it too, not just the
    # non-robust baseline above. Metaheuristics only -- see module docstring.
    robust_rows = run_robust_backend_benchmarks(explorer, ROBUST_BACKENDS, targets,
                                                n_samples=n_samples)

    nsga_row = run_nsga_benchmark(explorer, pop_size=pop_size, n_gen=n_gen, robust=False)
    nsga_robust_row = run_nsga_benchmark(explorer, pop_size=pop_size, n_gen=n_gen, robust=True)
    add_shared_hypervolume(nsga_row, nsga_robust_row)

    md = render_markdown(rows, nsga_row, n_samples, flow_available=explorer.amortized is not None,
                         age_rows=age_rows, robust_rows=robust_rows, nsga_robust_row=nsga_robust_row)

    if args.quick:
        print(md)
    else:
        with open(DOC_PATH, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"Wrote {DOC_PATH} ({len(rows)} backend rows"
              + (f", {len(robust_rows)} robust rows" if robust_rows else "")
              + (", 1 NSGA row" if nsga_row else "")
              + (" + 1 robust NSGA row" if nsga_robust_row else "") + ").")


if __name__ == "__main__":
    main()
