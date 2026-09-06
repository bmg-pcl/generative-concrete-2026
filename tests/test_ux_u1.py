"""
R8.6 WP-U1 acceptance tests: tab reorder, sidebar guide auto-collapse, and the
Calibration status line. Uses streamlit.testing.v1.AppTest, same pattern as
tests/test_app_smoke.py (headless, no browser).
"""
import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")

EXPECTED_TAB_ORDER = [
    "Config", "Compare Mixes", "Inverse Design", "Pareto Optimization",
    "Calibration", "Workflow", "Technical Report", "References",
]


def test_tabs_render_in_new_order():
    at = AppTest.from_file(APP, default_timeout=240).run()
    assert not at.exception, f"App raised: {at.exception}"

    labels = [t.label for t in at.tabs]
    assert labels == EXPECTED_TAB_ORDER


def test_calibration_status_text_present_no_overlay():
    """No overlay file is committed to the repo, so the tab must show the
    "no calibration data" sentence from the spec, verbatim."""
    at = AppTest.from_file(APP, default_timeout=240).run()
    assert not at.exception

    infos = [i.value for i in at.info]
    assert any(
        "No calibration data loaded — the model is running on the base UIUC "
        "dataset alone." in text
        for text in infos
    ), f"Calibration status text not found in info elements: {infos}"


def test_sidebar_guide_expanded_first_run_collapsed_after():
    """R8.6 WP-U1 deliverable 3: the sidebar's Step-by-step guide expander is
    expanded on the session's first script run and collapsed on every run after."""
    at = AppTest.from_file(APP, default_timeout=240).run()
    assert not at.exception

    guide = next(e for e in at.sidebar.expander if e.label == "Step-by-step guide")
    assert guide.proto.expanded is True

    # Any rerun (e.g. touching an unrelated widget) is a second script run.
    at.slider(key="cement_A").set_value(at.slider(key="cement_A").value).run()
    assert not at.exception

    guide2 = next(e for e in at.sidebar.expander if e.label == "Step-by-step guide")
    assert guide2.proto.expanded is False


def test_workflow_tab_drops_step_by_step_duplication():
    """Deliverable 2: the Workflow tab keeps the "three questions, three tools"
    table and decision guidance but no longer duplicates the sidebar's numbered
    step-by-step walkthrough."""
    at = AppTest.from_file(APP, default_timeout=240).run()
    assert not at.exception

    workflow_text = "\n".join(m.value for m in at.markdown)
    assert "Three questions, three tools" in workflow_text
    assert "How the amortized flow and NSGA fit together" in workflow_text
    # The dropped section's own heading must not survive.
    assert "The workflow, in order" not in workflow_text
