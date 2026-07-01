"""
Headless integration smoke test for the Streamlit app.

Uses streamlit.testing.v1.AppTest to run app.py end to end without a browser.
Streamlit executes every `with tab:` block on each run, so a clean run exercises
the code path of *every* tab -- catching runtime errors (like a bad backend route
or a stale reference) that module-level unit tests miss. Skipped if Streamlit is
not installed.
"""
import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def test_app_runs_without_exception():
    # One clean run executes every tab's code path; the advanced carbon tier is
    # covered separately by the fast ui_logic unit tests.
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, f"App raised: {at.exception}"
