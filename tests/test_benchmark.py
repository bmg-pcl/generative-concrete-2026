"""Fast smoke test for the backend benchmark script (scripts/benchmark_backends.py).

Exercises the plumbing with a tiny budget — it does NOT assert timing or accuracy
values (the metaheuristics are stochastic and machine-dependent). It only checks that
the runner returns rows with the documented columns and that the markdown renders.
The full benchmark is not part of CI (timing-sensitive).
"""
import pytest

from scripts.benchmark_backends import (
    BENCH_COLUMNS,
    ROBUST_BENCH_COLUMNS,
    ROBUST_BACKENDS,
    NSGA_COLUMNS,
    ROBUST_NSGA_COLUMNS,
    run_backend_benchmarks,
    run_robust_backend_benchmarks,
    run_nsga_benchmark,
    add_shared_hypervolume,
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


# --- R7.3a: robust-mode benchmarking --------------------------------------------
def test_robust_backend_excludes_flow():
    """The flow does not recondition on robust= inside sample_posterior (robust
    handling for the flow is a downstream filter in recommend_recipe), so a 'robust
    flow' row would misrepresent what ran. ROBUST_BACKENDS must not include it."""
    assert "flow" not in ROBUST_BACKENDS
    assert set(ROBUST_BACKENDS) == {"ga", "aco"}


def test_robust_backend_benchmark_rows_have_expected_columns():
    explorer = BayesFlowExplorer()
    rows = run_robust_backend_benchmarks(explorer, ["ga", "aco"], [45.0], n_samples=30, seed=0)
    assert len(rows) == 2
    for row in rows:
        assert set(ROBUST_BENCH_COLUMNS).issubset(row.keys())
        assert row["backend"] in ("ga", "aco")
        assert row["lower_bound_mae_MPa"] >= 0.0
        assert 0.0 <= row["in_support_pct"] <= 100.0
        # The robust row must NOT carry the non-robust metric name (median MAE would
        # misrepresent what the robust objective was actually scored on).
        assert "mae_MPa" not in row


def test_render_markdown_robust_section():
    robust_rows = [{"backend": "ga", "target_MPa": 45.0, "lower_bound_mae_MPa": 2.0,
                    "cloud_spread": 15.0, "in_support_pct": 95.0, "time_s": 1.0}]
    rows = [{"backend": "ga", "target_MPa": 45.0, "mae_MPa": 1.0,
             "cloud_spread": 20.0, "in_support_pct": 50.0, "time_s": 0.5}]
    md = render_markdown(rows, None, n_samples=30, flow_available=False, robust_rows=robust_rows)
    assert "## Robust mode" in md
    assert "| " + " | ".join(ROBUST_BENCH_COLUMNS) + " |" in md
    assert "lower_bound_mae_MPa" in md
    # The section must not claim a robust flow row ran.
    assert "robust flow row is shown" in md or "no robust flow row" in md


@pytest.mark.parametrize("robust", [False, True])
def test_nsga_benchmark_robust_flag(robust):
    pytest.importorskip("pymoo")
    explorer = BayesFlowExplorer()
    row = run_nsga_benchmark(explorer, pop_size=16, n_gen=4, robust=robust)
    assert row is not None
    assert set(NSGA_COLUMNS).issubset(row.keys())
    assert row["front_size"] >= 1
    assert "_strength" in row and "_carbon" in row and "_cost" in row


def test_add_shared_hypervolume():
    pytest.importorskip("pymoo")
    row_a = {"_strength": [40.0, 45.0], "_carbon": [200.0, 210.0], "_cost": [80.0, 85.0]}
    row_b = {"_strength": [42.0], "_carbon": [190.0], "_cost": [78.0]}
    add_shared_hypervolume(row_a, row_b)
    assert row_a["hypervolume"] >= 0.0
    assert row_b["hypervolume"] >= 0.0


def test_add_shared_hypervolume_noop_on_missing_row():
    row_a = {"_strength": [40.0], "_carbon": [200.0], "_cost": [80.0]}
    add_shared_hypervolume(row_a, None)
    assert "hypervolume" not in row_a


def test_render_markdown_combined_nsga_table_with_hypervolume():
    nsga_row = {"algorithm": "NSGA-II", "front_size": 12, "mean_strength_MPa": 55.0,
                "in_support_pct": 40.0, "time_s": 0.1, "hypervolume": 1000.0}
    nsga_robust_row = {"algorithm": "NSGA-II", "front_size": 8, "mean_strength_MPa": 52.0,
                       "in_support_pct": 100.0, "time_s": 0.2, "hypervolume": 700.0}
    rows = [{"backend": "ga", "target_MPa": 45.0, "mae_MPa": 1.0,
             "cloud_spread": 20.0, "in_support_pct": 50.0, "time_s": 0.5}]
    md = render_markdown(rows, nsga_row, n_samples=30, flow_available=False,
                         nsga_robust_row=nsga_robust_row)
    assert "| " + " | ".join(ROBUST_NSGA_COLUMNS) + " |" in md
    assert "non-robust" in md and "(robust)" in md
    assert "1000.0" in md and "700.0" in md
