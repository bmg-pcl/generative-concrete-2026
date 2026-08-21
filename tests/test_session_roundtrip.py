"""R5.1/R7.4a — session export/import round-trip guarantee.

The R3.4 regression (v1 exports silently dropped carbon_factors) shipped because no
test asserted values survive a round trip. These tests do, at three levels:

1. Pure: export_session/apply_session round-trip on a plain dict (no Streamlit run).
2. Symmetry: the export writes exactly version + SESSION_FIELDS — a field added to
   one side but not the other fails here, at edit time.
3. Integration (AppTest): mutated widget state -> export -> fresh session -> import ->
   the widgets and derived dicts show the imported values. This specifically covers
   the keyed-widget trap: cf_<mat>, mix sliders, and (R7.4a) the Config-tab cfg_<...>
   controls all retain their own widget state, so an import that only replaced the
   backing dicts would be overwritten on the next rerun.

R7.4a extends the guarantee to the ten Config-tab controls (transport, clinker
source, robust/age toggles, ...) that were previously lost on every export/import —
the "full session restore" item the R3.4 spec deferred.
"""
import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from ui.state import (  # noqa: E402
    SESSION_FIELDS,
    SLIDER_SPECS,
    DEFAULT_MIX_A,
    DEFAULT_MIX_B,
    CONFIG_DEFAULTS,
    export_session,
    apply_session,
)
from src.chemistry_simple import UNIT_COSTS, CARBON_FACTORS  # noqa: E402
from src.exotics import EXOTIC_ADMIXTURES  # noqa: E402

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def _fresh_state():
    """A plain-dict stand-in for st.session_state with the app's initial values."""
    state = {}
    for slot, default in (("A", DEFAULT_MIX_A), ("B", DEFAULT_MIX_B)):
        for (p, *_), v in zip(SLIDER_SPECS, default):
            state[f"{p}_{slot}"] = int(v)
    state["costs"] = UNIT_COSTS.copy()
    state["carbon_factors"] = CARBON_FACTORS.copy()
    for mat, val in state["carbon_factors"].items():
        state[f"cf_{mat}"] = float(val)
    state["exotic_a"] = {k: v["default"] for k, v in EXOTIC_ADMIXTURES.items()}
    state["exotic_b"] = {k: v["default"] for k, v in EXOTIC_ADMIXTURES.items()}
    state["epds"] = {}
    state.update(CONFIG_DEFAULTS)
    return state


def test_export_import_key_symmetry():
    exported = export_session(_fresh_state())
    assert set(exported.keys()) == {"version"} | set(SESSION_FIELDS)


def test_pure_round_trip_every_field():
    src = _fresh_state()
    # Mutate one value in every exported field.
    src["cement_A"] = 444
    src["water_B"] = 199
    src["costs"]["cement"] = 0.99
    src["carbon_factors"]["cement"] = 0.5
    src["cf_cement"] = 0.5
    src["exotic_a"]["silica_fume"] = 25
    src["exotic_b"]["nano_silica"] = 2
    src["epds"] = {"cement": {"value": 0.55, "reference": "EPD-XYZ-2025"}}
    # R7.4a: mutate a representative sample of the Config-tab controls too.
    src["cfg_transport_km"] = 150
    src["cfg_robust"] = False
    src["cfg_fix_age"] = False
    src["cfg_design_age"] = 90
    src["cfg_kiln_fuel"] = "natural_gas"
    src["cfg_electricity"] = "hydro"
    src["cfg_capture_rate"] = 0.9
    src["cfg_waste_factor"] = 0.08  # R8.0 WP-A A2
    src["cfg_site_temp_c"] = 32.0        # R8.0 WP-E Decision 1
    src["cfg_transport_detail"] = True   # R8.0 WP-E Decision 2

    exported = export_session(src)
    dst = _fresh_state()
    apply_session(exported, dst)

    assert dst["cement_A"] == 444
    assert dst["water_B"] == 199
    assert dst["costs"]["cement"] == 0.99
    assert dst["carbon_factors"]["cement"] == 0.5
    assert dst["cf_cement"] == 0.5, "import must write the cf_ widget keys too"
    assert dst["exotic_a"]["silica_fume"] == 25
    assert dst["exotic_b"]["nano_silica"] == 2
    assert dst["epds"]["cement"]["reference"] == "EPD-XYZ-2025"
    assert dst["cfg_transport_km"] == 150
    assert dst["cfg_robust"] is False
    assert dst["cfg_fix_age"] is False
    assert dst["cfg_design_age"] == 90
    assert dst["cfg_kiln_fuel"] == "natural_gas"
    assert dst["cfg_electricity"] == "hydro"
    assert dst["cfg_capture_rate"] == 0.9
    assert dst["cfg_waste_factor"] == 0.08
    assert dst["cfg_site_temp_c"] == 32.0
    assert dst["cfg_transport_detail"] is True
    # And the whole export re-exports identically (idempotent round trip).
    assert export_session(dst) == exported


def test_config_defaults_all_present_in_a_fresh_state():
    """Every CONFIG_DEFAULTS key must round-trip -- a control added to CONFIG_DEFAULTS
    but never read by export_session (or vice versa) would silently drop data."""
    exported = export_session(_fresh_state())
    assert set(exported["ui_config"].keys()) == set(CONFIG_DEFAULTS.keys())


def test_apptest_round_trip_survives_widget_state():
    # Session 1: mutate a keyed slider, a keyed emission factor, AND (R7.4a) two
    # Config-tab controls (a number_input and a toggle) via the real widgets.
    at = AppTest.from_file(APP, default_timeout=240).run()
    at.slider(key="cement_A").set_value(444).run()
    at.number_input(key="cf_cement").set_value(0.5).run()
    at.number_input(key="cfg_transport_km").set_value(150).run()
    default_robust = at.toggle(key="cfg_robust").value
    at.toggle(key="cfg_robust").set_value(not default_robust).run()
    at.slider(key="cfg_waste_factor").set_value(0.08).run()  # R8.0 WP-A A2
    at.number_input(key="cfg_site_temp_c").set_value(32.0).run()   # R8.0 WP-E Decision 1
    default_transport_detail = at.toggle(key="cfg_transport_detail").value
    at.toggle(key="cfg_transport_detail").set_value(not default_transport_detail).run()  # Decision 2
    exported = export_session(at.session_state)
    assert exported["mix_a"][0] == 444
    assert exported["carbon_factors"]["cement"] == 0.5
    assert exported["ui_config"]["cfg_transport_km"] == 150
    assert exported["ui_config"]["cfg_robust"] == (not default_robust)
    assert exported["ui_config"]["cfg_waste_factor"] == 0.08
    assert exported["ui_config"]["cfg_site_temp_c"] == 32.0
    assert exported["ui_config"]["cfg_transport_detail"] == (not default_transport_detail)

    # Session 2 (fresh): widgets have their own (default) state; the import must win.
    at2 = AppTest.from_file(APP, default_timeout=240).run()
    assert at2.slider(key="cement_A").value != 444
    assert at2.number_input(key="cfg_transport_km").value != 150
    assert at2.slider(key="cfg_waste_factor").value != 0.08
    assert at2.number_input(key="cfg_site_temp_c").value != 32.0
    apply_session(exported, at2.session_state)
    at2.run()
    assert not at2.exception
    assert at2.slider(key="cement_A").value == 444
    assert at2.number_input(key="cf_cement").value == 0.5
    assert at2.number_input(key="cfg_transport_km").value == 150
    assert at2.toggle(key="cfg_robust").value == (not default_robust)
    assert at2.slider(key="cfg_waste_factor").value == 0.08
    assert at2.number_input(key="cfg_site_temp_c").value == 32.0
    assert at2.toggle(key="cfg_transport_detail").value == (not default_transport_detail)
    # The derived dict must agree after the rerun (the keyed widgets did not
    # overwrite the imported factor — the exact live-session bug this guards).
    assert at2.session_state["carbon_factors"]["cement"] == 0.5
