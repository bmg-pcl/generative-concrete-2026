"""
R8.6 WP-U3 acceptance tests: Pareto tab run feedback, results layout, chart
legibility. Uses streamlit.testing.v1.AppTest (see tests/test_app_smoke.py for
the pattern) -- headless, no browser, no live GA run (too slow for a unit
test; the run path is instead verified by hand, see the WP report).
"""
import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from ui.pareto import RESULTS_PLACEHOLDER_TEXT  # noqa: E402

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def test_app_runs_clean():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, f"App raised: {at.exception}"


def test_prerun_placeholder_present_in_results_area():
    """Deliverable 1: the results column is never blank pre-run -- a bordered
    container carries the verbatim placeholder text before any run starts."""
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception

    markdown_texts = [m.value for m in at.markdown]
    assert any(RESULTS_PLACEHOLDER_TEXT in t for t in markdown_texts), (
        "Pre-run placeholder text not found in the Pareto tab's results area."
    )


def test_no_bare_best_fitness_metric_preruns():
    """Deliverable 3: the headline 'Best Fitness' metric card is gone entirely
    (replaced by the three physical metrics); nothing on the pre-run page
    presents a bare unitless fitness number as a metric card."""
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception

    labels = [m.label for m in at.metric]
    assert "Best Fitness" not in labels, (
        "A bare 'Best Fitness' metric card is still present -- deliverable 3 "
        "requires replacing it with the three physical metrics."
    )


def test_philosophical_note_moved_into_collapsed_expander_verbatim():
    """Deliverable 7: the four-paragraph 'Philosophical Note on Stochastic
    Search' essay is collapsed inside an expander titled like 'How the
    optimizers work', with its text preserved verbatim (spot-check a phrase
    from each paragraph, including the exact 'Violin Plots' wording that
    deliverable 6 must stay consistent with the gene-pool chart)."""
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception

    note_expanders = [e for e in at.expander if "How the optimizers work" in e.label]
    assert len(note_expanders) == 1, "Expected exactly one 'How the optimizers work' expander."
    exp = note_expanders[0]

    text = "\n".join(m.value for m in exp.markdown)
    for phrase in [
        "Philosophical Note on Stochastic Search",
        "non-convex and high-dimensional",
        "Genetic Algorithm (GA) Mechanics",
        "Premature Convergence",
        "Violin Plots",
        "Simulated Annealing (SA) Mechanics",
        "Pareto Frontier",
        "non-dominated",
    ]:
        assert phrase in text, f"Essay text missing expected phrase: {phrase!r}"


def test_objective_weights_caption_present():
    """Deliverable 4: one caption under 'Objective Weights' truthfully
    describing what the sliders multiply (read from scalarized_fitness)."""
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception

    captions = [c.value for c in at.caption]
    assert any("multiply the optimizer's raw predicted quantities" in c for c in captions), (
        "Objective Weights caption not found or wording changed unexpectedly."
    )


def test_generative_and_nsga_unaffected_by_import():
    """Cheap sanity check that importing ui.pareto doesn't blow up and that
    the optimizer-facing helpers it re-exports/uses are untouched imports
    (no behavior change -- src/ga.py, src/nsga.py are not owned by this WP
    and are proven green separately by tests/test_generative.py and
    tests/test_nsga.py per the gate command)."""
    import ui.pareto as pareto_mod
    from src.ga import GeneticOptimizer
    from src.nsga import run_nsga

    assert pareto_mod.GeneticOptimizer is GeneticOptimizer
    assert pareto_mod.run_nsga is run_nsga
