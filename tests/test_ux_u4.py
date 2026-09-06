"""
AppTest coverage for WP-U4 (R8.6): Inverse Design tab recipe presentation and
robust-mode framing.

Pattern: tests/test_app_smoke.py (streamlit.testing.v1.AppTest, full headless
app run). Streamlit executes every `with tab:` block on each script run, so a
plain full-app run already drives a real recommended-recipe search on the
Inverse Design tab with the default widget values (target=45 MPa, backend=
"auto", robust=True) -- no button click or extra driving is needed to reach
`rec is not None` and exercise the new presentation code.
"""
import os
import re

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def _recipe_dataframe(at):
    """Find the recommended-recipe table among every rendered dataframe (the
    Compare/Calibration/Pareto tabs render their own, with different columns)
    by its distinctive column set."""
    for df_el in at.dataframe:
        if list(df_el.value.columns) == ["Material", "Amount"]:
            return df_el.value
    return None


def _predicted_strength(at):
    labels = [m.label for m in at.metric]
    idx = labels.index("Predicted Strength")
    m = re.match(r"[-+]?\d+\.?\d*", at.metric[idx].value)
    assert m, f"unexpected 'Predicted Strength' metric value: {at.metric[idx].value!r}"
    return float(m.group())


def test_app_runs_clean_and_recipe_renders_as_a_table():
    """Deliverable 1: the recommended recipe is a compact table (kg/m3 per
    material + age), not the old run-on text line of eight numbers."""
    at = AppTest.from_file(APP, default_timeout=240).run()
    assert not at.exception

    assert any("Recipe (kg/m³, age in days)" in m.value for m in at.markdown)

    df = _recipe_dataframe(at)
    assert df is not None, "recommended-recipe table not found among rendered dataframes"
    assert len(df) == 8  # 7 materials + age (PARAM_NAMES order, via SLIDER_SPECS)
    materials = set(df["Material"])
    assert {"Cement", "Water", "Age (days)"} <= materials


def test_load_buttons_present_side_by_side_with_original_keys_and_callbacks():
    """Deliverable 3: same keys/callbacks as before (keyed-widget rule in
    ui/state.py -- the on_click write must keep landing before the Compare
    tab's sliders reinstantiate), just laid out adjacently under the recipe."""
    at = AppTest.from_file(APP, default_timeout=240).run()
    assert not at.exception

    btn_a = at.button(key="load_rec_a")
    btn_b = at.button(key="load_rec_b")
    assert btn_a.label == "Load into Mix A"
    assert btn_b.label == "Load into Mix B"

    # Exercise the callback end to end, exactly as test_app_smoke.py's
    # test_mix_slider_state_propagates does for the Mix A path: clicking must
    # still write the keyed cement slider (the value-propagation contract this
    # WP is explicitly forbidden from changing).
    df = _recipe_dataframe(at)
    expected_cement = int(df.loc[df["Material"] == "Cement", "Amount"].iloc[0].split()[0])
    at.button(key="load_rec_a").click().run()
    assert not at.exception
    assert at.slider(key="cement_A").value == expected_cement
    assert at.session_state["cement_A"] == expected_cement


def test_robust_framing_caption_is_truthful_and_conditional():
    """Deliverable 2: when robust mode is ON (CONFIG_DEFAULTS default) and the
    point estimate exceeds the target, an adjacent caption must explain that
    the *guaranteed lower bound* is what met the target and the point
    estimate is higher by design -- never presented as an overshoot. The
    caption must be absent whenever that condition doesn't hold, and must
    disappear entirely when robust mode is off."""
    at = AppTest.from_file(APP, default_timeout=240).run()
    assert not at.exception
    assert at.toggle(key="cfg_robust").value is True  # CONFIG_DEFAULTS: robust ON by default

    strength = _predicted_strength(at)
    target = 45.0  # ui/inverse.py's st.number_input default
    captions = [c.value for c in at.caption]
    has_framing = any("guaranteed lower bound" in c for c in captions)
    if strength > target:
        assert has_framing, (
            f"robust mode is on and strength {strength} > target {target}, "
            "but no robust-framing caption was rendered"
        )
    else:
        assert not has_framing

    # Robust OFF: no change per spec -- the framing caption must never appear.
    at.toggle(key="cfg_robust").set_value(False).run()
    assert not at.exception
    captions_off = [c.value for c in at.caption]
    assert not any("guaranteed lower bound" in c for c in captions_off)
