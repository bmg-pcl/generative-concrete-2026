"""Fast smoke test for the backend benchmark script (scripts/benchmark_backends.py).

Exercises the plumbing with a tiny budget — it does NOT assert timing or accuracy
values (the metaheuristics are stochastic and machine-dependent). It only checks that
the runner returns rows with the documented columns and that the markdown renders.
The full benchmark is not part of CI (timing-sensitive).
"""

from scripts.benchmark_backends import (
    BENCH_COLUMNS,
    NSGA_COLUMNS,
    run_backend_benchmarks,
    render_markdown,
)
from src.bayesian import BayesFlowExplorer


def test_backend_benchmark_rows_have_expected_columns():
    explorer = BayesFlowExplorer()
    # GA + ACO are always available (no trained flow needed); tiny cloud, one target.
    rows = run_backend_benchmarks(explorer, ["ga", "aco"], [45.0], n_samples=30, seed=0)
    assert len(rows) == 2
    for row in rows:
        assert set(BENCH_COLUMNS).issubset(row.keys())
        assert row["backend"] in ("ga", "aco")
        assert row["mae_MPa"] >= 0.0
        assert 0.0 <= row["in_support_pct"] <= 100.0


def test_flow_skipped_when_untrained():
    explorer = BayesFlowExplorer()
    # Force the "no trained flow" branch regardless of the local environment.
    explorer._amortized = False
    rows = run_backend_benchmarks(explorer, ["flow", "ga"], [45.0], n_samples=20, seed=0)
    assert all(r["backend"] != "flow" for r in rows), "flow must be skipped when untrained"
    assert any(r["backend"] == "ga" for r in rows)


def test_render_markdown_has_tables():
    rows = [{"backend": "ga", "target_MPa": 45.0, "mae_MPa": 1.0,
             "cloud_spread": 20.0, "in_support_pct": 50.0, "time_s": 0.5}]
    nsga_row = {"algorithm": "NSGA-II", "front_size": 12,
                "mean_strength_MPa": 55.0, "in_support_pct": 40.0, "time_s": 0.1}
    md = render_markdown(rows, nsga_row, n_samples=30, flow_available=False)
    assert "# Backend benchmarks" in md
    assert "| " + " | ".join(BENCH_COLUMNS) + " |" in md
    assert "| " + " | ".join(NSGA_COLUMNS) + " |" in md
    assert "not benchmarked here" in md  # flow-skipped note
    assert "python -m scripts.benchmark_backends" in md
