"""
Headless integration smoke test for the Streamlit app.

Uses streamlit.testing.v1.AppTest to run app.py end to end without a browser.
Streamlit executes every `with tab:` block on each run, so a clean run exercises
the code path of *every* tab -- catching runtime errors (like a bad backend route
or a stale reference) that module-level unit tests miss. Skipped if Streamlit is
not installed.
"""
import os

import numpy as np
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def test_app_runs_without_exception():
    # One clean run executes every tab's code path; the advanced carbon tier is
    # covered separately by the fast ui_logic unit tests.
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, f"App raised: {at.exception}"


def test_mix_slider_state_propagates():
    """R4.1 value-propagation gate (not a crash-only smoke): a programmatic load must
    change the keyed slider's value AND the Mix-A strength metric that derives from it.

    This is the regression the earlier unkeyed sliders risked: writing new mix state and
    rerunning could leave the sliders (and metrics) showing the old mix."""
    at = AppTest.from_file(APP, default_timeout=240).run()
    assert not at.exception

    # Baseline cement value on the Mix A slider and the Mix A strength metric.
    cement0 = at.slider(key="cement_A").value
    strength0 = next(m for m in at.metric if m.label == "Strength").value

    # Simulate the "Load into Mix A" path: set the keyed slider to a distinctly
    # different value, then rerun. If keying/derivation is correct, both the slider
    # and the derived metric change.
    new_cement = 520 if cement0 < 400 else 150
    at.slider(key="cement_A").set_value(new_cement).run()
    assert not at.exception

    assert at.slider(key="cement_A").value == new_cement
    assert at.session_state["cement_A"] == new_cement
    strength1 = next(m for m in at.metric if m.label == "Strength").value
    assert strength1 != strength0, "Mix A strength metric did not track the slider change"


def test_exposure_compliance_widgets_render_and_verdict_shows():
    """R8.2 WP-3: the Config-tab pack/class selectors and the Compare tab's
    compliance panel + cross-jurisdiction table render without exception, both
    before and after a real selection is made. UNKNOWN/PASS/FAIL render via
    visually distinct elements (st.warning/st.success/st.error)."""
    at = AppTest.from_file(APP, default_timeout=240).run()
    assert not at.exception

    # Inert by default: no compliance verdict banner yet, but the headline
    # cross-jurisdiction table section is always present.
    assert any("Cross-jurisdiction" in h.value for h in at.subheader)
    assert not any("Compliance (A):" in s.value for s in at.success) \
        and not any("Compliance (A):" in s.value for s in at.error) \
        and not any("Compliance (A):" in s.value for s in at.warning)

    at.selectbox(key="cfg_exposure_pack").set_value("en206").run()
    assert not at.exception
    at.selectbox(key="cfg_exposure_class").set_value("XC4").run()
    assert not at.exception

    verdict_alerts = list(at.success) + list(at.error) + list(at.warning)
    verdict_texts = [a.value for a in verdict_alerts]
    assert any("Compliance (A): en206.XC4" in t for t in verdict_texts)
    assert any("Compliance (B): en206.XC4" in t for t in verdict_texts)
    # The advisory disclosure names the standard, on a real caption surface.
    assert any("EN 206" in c.value for c in at.caption)


def test_preset_load_callback_sets_sliders():
    """R4.1: the preset on_change callback writes the keyed sliders (the shared load
    mechanism). After selecting a dataset preset, the sliders reflect that preset."""
    from src.data_fetcher import load_data

    at = AppTest.from_file(APP, default_timeout=240).run()
    assert not at.exception

    options = at.selectbox(key="preset_A").options
    preset_label = next(o for o in options if o != "Custom")
    at.selectbox(key="preset_A").set_value(preset_label).run()
    assert not at.exception

    # The label is "Dataset #<row> (<strength> MPa measured)"; the cement value is
    # column 0 of that dataset row.
    row = int(preset_label.split("#")[1].split(" ")[0])
    # Mirror load_mix_into's clamp+round so the expectation matches the widget value.
    raw_cement = float(load_data().iloc[row].values[0])
    expected_cement = int(np.clip(round(raw_cement), 100, 550))
    assert at.slider(key="cement_A").value == expected_cement
