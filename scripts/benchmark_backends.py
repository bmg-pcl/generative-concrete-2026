"""Benchmark the inverse-design backends (flow / GA / ACO) and one NSGA run.

For a small grid of target strengths, each metaheuristic/flow backend samples a mix
cloud; we measure how well it hits the target, how spread out it is, how much of it
stays inside the trusted data region, and how long it took. NSGA is reported
separately (it maps a whole front rather than one target).

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
BENCH_COLUMNS = ["backend", "target_MPa", "mae_MPa", "cloud_spread", "in_support_pct", "time_s"]
NSGA_COLUMNS = ["algorithm", "front_size", "mean_strength_MPa", "in_support_pct", "time_s"]
DOC_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "BENCHMARKS.md")


def _cloud_metrics(explorer, samples, target):
    predictor = explorer.predictor
    strength = predictor.predict_batch(samples)
    return {
        "mae_MPa": round(float(np.mean(np.abs(strength - target))), 2),
        "cloud_spread": round(float(np.mean(np.std(samples, axis=0))), 2),
        "in_support_pct": round(float(np.mean(predictor.in_support(samples)) * 100.0), 1),
    }


def run_backend_benchmarks(explorer, backends, targets, n_samples=1500, seed=0):
    """Return a list of per-(backend, target) metric rows.

    Backends not currently available (e.g. 'flow' with no trained weights) are
    skipped; the returned rows say which ran, so a caller can note the skip."""
    rows = []
    for method in backends:
        if method == "flow" and explorer.amortized is None:
            continue  # no trained flow weights — skip rather than error
        for t in targets:
            np.random.seed(seed)  # best-effort determinism for the metaheuristics
            start = time.time()
            samples = explorer.sample_posterior(float(t), n_samples=n_samples, method=method)
            elapsed = time.time() - start
            row = {"backend": method, "target_MPa": float(t), "time_s": round(elapsed, 2)}
            row.update(_cloud_metrics(explorer, samples, t))
            rows.append(row)
    return rows


def run_nsga_benchmark(explorer, pop_size=60, n_gen=30, seed=1):
    """Return a single NSGA metrics row, or None if pymoo is unavailable."""
    try:
        from src.nsga import run_nsga
    except Exception:
        return None
    predictor = explorer.predictor
    start = time.time()
    try:
        out = run_nsga(predictor, algorithm="nsga2", pop_size=pop_size, n_gen=n_gen,
                       random_seed=seed)
    except ImportError:
        return None
    elapsed = time.time() - start
    return {
        "algorithm": out["algorithm"],
        "front_size": int(out["front_size"]),
        "mean_strength_MPa": round(float(np.mean(out["strength"])), 1),
        "in_support_pct": round(float(np.mean(predictor.in_support(out["mixes"])) * 100.0), 1),
        "time_s": round(elapsed, 2),
    }


def _md_table(columns, rows):
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(r[c]) for c in columns) + " |" for r in rows]
    return "\n".join([header, sep, *body])


def render_markdown(rows, nsga_row, n_samples, flow_available):
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
    if nsga_row is not None:
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
    rows = run_backend_benchmarks(explorer, DEFAULT_BACKENDS, targets, n_samples=n_samples)
    nsga_row = run_nsga_benchmark(explorer, pop_size=20 if args.quick else 60,
                                  n_gen=8 if args.quick else 30)
    md = render_markdown(rows, nsga_row, n_samples, flow_available=explorer.amortized is not None)

    if args.quick:
        print(md)
    else:
        with open(DOC_PATH, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"Wrote {DOC_PATH} ({len(rows)} backend rows"
              + (", 1 NSGA row" if nsga_row else "") + ").")


if __name__ == "__main__":
    main()
