"""R5.2 — headless CLI tests (src/cli.py).

The CLI is wiring, not logic: predict must equal compute_metrics called directly,
design must track the target like recommend_recipe does, and the module must never
import Streamlit (it is the seam for batch/procurement use).
"""
import json
import subprocess
import sys

import numpy as np
import pytest

from src.cli import (main, load_mix, load_project_config, CliError, validate_clinker_source,
                     validate_waste_factor, DEFAULT_RUN_CONFIG)
from src.models import StrengthPredictor
from src.ui_logic import PARAM_NAMES, compute_metrics

MIX = {"cement": 350, "slag": 100, "ash": 0, "water": 175, "superplasticizer": 5,
       "coarse_agg": 1000, "fine_agg": 750, "age": 28}


@pytest.fixture()
def mix_file(tmp_path):
    p = tmp_path / "mix.json"
    p.write_text(json.dumps(MIX))
    return str(p)


@pytest.fixture()
def config_file(tmp_path):
    p = tmp_path / "project.json"
    p.write_text(json.dumps({
        "costs": {"cement": 0.20},
        "carbon_factors": {"cement": 0.85},
        "config": {"transport_km": 50.0, "robust": True},
    }))
    return str(p)


def test_predict_matches_compute_metrics(mix_file, config_file, capsys):
    rc = main(["predict", "--mix", mix_file, "--config", config_file])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)

    cfg = load_project_config(config_file)
    mix, exotic = load_mix(mix_file)
    expected = compute_metrics(
        mix, exotic, cfg["costs"], StrengthPredictor(),
        carbon_kwargs={"transport_km": 50.0, "cement_type": "OPC",
                       "factors": cfg["carbon_factors"]},
    )
    for key in ("strength", "carbon", "cost", "interval_lo", "interval_hi", "curing"):
        assert out[key] == pytest.approx(expected[key], rel=1e-6), key
    # The overridden cost and factor must actually have taken effect.
    assert cfg["costs"]["cement"] == 0.20
    assert out["mix"] == {k: float(v) for k, v in MIX.items()}


def test_predict_vector_mix_and_ticket(tmp_path, capsys):
    vec = [float(MIX[p]) for p in PARAM_NAMES]
    mixp = tmp_path / "vec.json"
    mixp.write_text(json.dumps(vec))
    ticket = tmp_path / "ticket.csv"
    rc = main(["predict", "--mix", str(mixp), "--ticket", str(ticket)])
    assert rc == 0
    lines = ticket.read_text().splitlines()
    assert lines[0] == "section,key,value"
    assert any(line.startswith("mix,cement,") for line in lines)
    assert any(line.startswith("carbon_kgCO2,TOTAL,") for line in lines)


def test_design_tracks_target(tmp_path, capsys):
    # Default config is ROBUST mode: the designer matches the conformal LOWER bound
    # to the target (the mean overshoots by the interval width, by design).
    rc = main(["design", "--target", "45", "--backend", "ga", "--age", "28"])
    assert rc == 0
    rec = json.loads(capsys.readouterr().out)
    assert abs(rec["interval_lo"] - 45.0) < 6.0
    # R7.1: point prediction and interval come from one distribution, so the point
    # can never cross its own bound — restored now that it is guaranteed (was
    # unassertable under the old two-model architecture; see docs/PAPER.md §8.3).
    assert rec["interval_lo"] <= rec["strength"] <= rec["interval_hi"]
    assert rec["params"]["age"] == pytest.approx(28.0, abs=1.0)
    assert set(PARAM_NAMES) <= set(rec["params"])

    # With robust off, the point (median) prediction tracks the target.
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"config": {"robust": False}}))
    rc = main(["design", "--target", "45", "--backend", "ga", "--age", "28",
               "--config", str(cfg)])
    assert rc == 0
    rec = json.loads(capsys.readouterr().out)
    assert abs(rec["strength"] - 45.0) < 6.0


def test_pareto_front_csv(tmp_path, capsys):
    pytest.importorskip("pymoo")
    out = tmp_path / "front.csv"
    rc = main(["pareto", "--pop", "24", "--gen", "6", "--out", str(out)])
    assert rc == 0
    lines = out.read_text().strip().splitlines()
    assert lines[0] == ",".join(list(PARAM_NAMES) + ["strength", "carbon", "cost"])
    assert len(lines) >= 2
    assert len(lines[1].split(",")) == len(PARAM_NAMES) + 3


def test_input_errors_exit_1(tmp_path, capsys):
    bad_mix = tmp_path / "bad.json"
    bad_mix.write_text(json.dumps({"cement": 300}))  # missing 7 params
    assert main(["predict", "--mix", str(bad_mix)]) == 1
    assert "missing parameter" in capsys.readouterr().err

    bad_cfg = tmp_path / "cfg.json"
    bad_cfg.write_text(json.dumps({"config": {"unknown_option": 1}}))
    mixp = tmp_path / "mix.json"
    mixp.write_text(json.dumps(MIX))
    assert main(["predict", "--mix", str(mixp), "--config", str(bad_cfg)]) == 1


def test_load_mix_rejects_wrong_vector_length(tmp_path):
    p = tmp_path / "short.json"
    p.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(CliError):
        load_mix(str(p))


def test_cli_never_imports_streamlit():
    # Fresh interpreter so other tests' imports can't pollute sys.modules.
    code = "import src.cli, sys; assert 'streamlit' not in sys.modules, 'CLI pulled in streamlit'"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_epd_plugs_into_cli(tmp_path, capsys):
    """--epd attaches supplier factors: carbon drops accordingly and the ticket
    discloses the EPD as the source of the cement factor."""
    mixp = tmp_path / "mix.json"
    mixp.write_text(json.dumps(MIX))
    epdp = tmp_path / "epds.json"
    epdp.write_text(json.dumps(
        {"epds": {"cement": {"value": 0.55, "reference": "EPD-BREVIK-2025"}}}))
    ticket = tmp_path / "ticket.csv"

    assert main(["predict", "--mix", str(mixp)]) == 0
    base = json.loads(capsys.readouterr().out)["carbon"]
    assert main(["predict", "--mix", str(mixp), "--epd", str(epdp),
                 "--ticket", str(ticket)]) == 0
    with_epd = json.loads(capsys.readouterr().out)["carbon"]
    assert with_epd == pytest.approx(base - MIX["cement"] * (0.912 - 0.55), rel=1e-6)
    assert 'provenance,cement,"epd:EPD-BREVIK-2025"' in ticket.read_text()

    # An EPD naming an unknown material is rejected (exit 1), not silently ignored.
    bad = tmp_path / "bad_epd.json"
    bad.write_text(json.dumps({"epds": {"unobtainium": {"value": 1.0}}}))
    assert main(["predict", "--mix", str(mixp), "--epd", str(bad)]) == 1


def test_config_uses_session_export_schema(tmp_path):
    # A real session export (with mix_a/mix_b etc.) must be accepted as-is.
    session = {"version": 2, "mix_a": [300] * 8, "mix_b": [300] * 8,
               "costs": {"cement": 0.33}, "carbon_factors": {"cement": 0.7},
               "exotic_a": {}, "exotic_b": {}}
    p = tmp_path / "gmd_session.json"
    p.write_text(json.dumps(session))
    cfg = load_project_config(str(p))
    assert cfg["costs"]["cement"] == 0.33
    assert cfg["carbon_factors"]["cement"] == 0.7
    assert np.isclose(cfg["run"]["transport_km"], 0.0)


def test_config_accepts_real_ui_session_export(tmp_path):
    """R7.4a regression guard: the UI's session export gained a nested "ui_config"
    block (the Config-tab widget state). The CLI's OWN "config" block is a distinct,
    semantic schema (advanced/robust/age/...) that rejects unknown keys -- if the two
    had collided on the same field name, feeding a real UI export to --config would
    raise CliError("Unknown config option(s): cfg_..."). It must not."""
    pytest.importorskip("streamlit")
    import ui.state as state_mod

    state = {}
    for slot, default in (("A", state_mod.DEFAULT_MIX_A), ("B", state_mod.DEFAULT_MIX_B)):
        for (p, *_), v in zip(state_mod.SLIDER_SPECS, default):
            state[f"{p}_{slot}"] = int(v)
    state["costs"] = state_mod.UNIT_COSTS.copy()
    state["carbon_factors"] = state_mod.CARBON_FACTORS.copy()
    state["exotic_a"], state["exotic_b"], state["epds"] = {}, {}, {}
    state.update(state_mod.CONFIG_DEFAULTS)

    exported = state_mod.export_session(state)
    assert "ui_config" in exported  # sanity: this is the field under test
    p = tmp_path / "gmd_session.json"
    p.write_text(json.dumps(exported))

    cfg = load_project_config(str(p))  # must not raise CliError
    assert cfg["run"] == {"advanced": False, "transport_km": 0.0, "cement_type": "OPC",
                          "robust": True, "age": None, "clinker_source": None,
                          "waste_factor": 0.0}


# --- R7.5 WP-4: clinker_source boundary validation --------------------------------
# `clinker_source`'s contents used to pass through unchecked into `FUEL_EF[fuel]` /
# `GRID_EF[elec]` inside `clinker_scope_split` (src/chemistry_advanced.py, frozen
# surface — not touched here). An unknown fuel or grid must be caught at the CLI
# boundary and turned into a clean exit-1 CliError, not an uncaught KeyError traceback.

def test_clinker_source_none_is_the_legitimate_default():
    assert validate_clinker_source(None) is None


def test_clinker_source_valid_descriptor_passes():
    assert validate_clinker_source({
        "kiln_fuel": "natural_gas", "electricity": "hydro",
        "capture": {"rate": 0.9, "energy_kwh_per_tCO2": 150},
    }) is None
    assert validate_clinker_source({}) is None  # every field optional


def test_clinker_source_must_be_a_dict():
    assert validate_clinker_source("coal") is not None
    assert validate_clinker_source(["coal"]) is not None


def test_clinker_source_unknown_fuel_names_valid_fuels():
    err = validate_clinker_source({"kiln_fuel": "hydrogen"})
    assert err is not None
    assert "hydrogen" in err
    # A user can't guess "alt_waste" -- the message must actually name it.
    assert "alt_waste" in err


def test_clinker_source_unknown_grid_names_valid_grids():
    err = validate_clinker_source({"electricity": "grid_ZA"})
    assert err is not None
    assert "grid_ZA" in err
    assert "grid_EU" in err


@pytest.mark.parametrize("rate", [1.5, -0.1, "high"])
def test_clinker_source_rejects_bad_capture_rate(rate):
    assert validate_clinker_source({"capture": {"rate": rate}}) is not None


@pytest.mark.parametrize("rate", [0.0, 0.5, 1.0])
def test_clinker_source_accepts_boundary_capture_rate(rate):
    assert validate_clinker_source({"capture": {"rate": rate}}) is None


def test_clinker_source_rejects_negative_capture_energy():
    assert validate_clinker_source(
        {"capture": {"energy_kwh_per_tCO2": -5}}) is not None
    assert validate_clinker_source(
        {"capture": {"energy_kwh_per_tCO2": 0}}) is None


def test_clinker_source_config_unknown_fuel_exits_1_no_traceback(tmp_path, capsys):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(
        {"config": {"clinker_source": {"kiln_fuel": "hydrogen"}}}))
    mixp = tmp_path / "mix.json"
    mixp.write_text(json.dumps(MIX))
    rc = main(["predict", "--mix", str(mixp), "--config", str(cfg)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "Error:" in err
    assert "hydrogen" in err
    assert "alt_waste" in err  # the valid fuels are actually named
    assert "Traceback" not in err


def test_clinker_source_config_unknown_grid_exits_1(tmp_path, capsys):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(
        {"config": {"clinker_source": {"electricity": "grid_ZA"}}}))
    mixp = tmp_path / "mix.json"
    mixp.write_text(json.dumps(MIX))
    rc = main(["predict", "--mix", str(mixp), "--config", str(cfg)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "grid_ZA" in err
    assert "Traceback" not in err


def test_clinker_source_config_bad_capture_rate_exits_1(tmp_path, capsys):
    mixp = tmp_path / "mix.json"
    mixp.write_text(json.dumps(MIX))
    for bad_rate in (1.5, -0.1):
        cfg = tmp_path / "cfg.json"
        cfg.write_text(json.dumps(
            {"config": {"clinker_source": {"kiln_fuel": "coal",
                                            "capture": {"rate": bad_rate}}}}))
        rc = main(["predict", "--mix", str(mixp), "--config", str(cfg)])
        assert rc == 1, bad_rate
        assert "Traceback" not in capsys.readouterr().err


def test_clinker_source_config_valid_descriptor_reaches_predict(tmp_path, capsys):
    """A valid clinker_source is accepted and actually used (not just permitted)."""
    mixp = tmp_path / "mix.json"
    mixp.write_text(json.dumps(MIX))
    base_cfg = tmp_path / "base_cfg.json"
    base_cfg.write_text(json.dumps({"config": {"advanced": True}}))
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"config": {
        "advanced": True,
        "clinker_source": {"kiln_fuel": "natural_gas", "electricity": "hydro",
                           "capture": {"rate": 0.9}},
    }}))
    rc_base = main(["predict", "--mix", str(mixp), "--config", str(base_cfg)])
    assert rc_base == 0
    base_carbon = json.loads(capsys.readouterr().out)["carbon"]

    rc = main(["predict", "--mix", str(mixp), "--config", str(cfg)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["carbon"] < base_carbon  # low-carbon clinker source lowers total carbon


# --- R8.0 WP-A A1/A2: ticket reconciliation & waste-factor boundary validation ----

def test_cli_ticket_reconciles_for_dosed_mix(tmp_path, capsys):
    """A1 headline gate via the CLI: for a mix with an "exotic" dosing dict, the
    written ticket's carbon_kgCO2,TOTAL row must equal the displayed carbon."""
    mix_with_exotic = dict(MIX)
    mix_with_exotic["exotic"] = {"silica_fume": 50}
    mixp = tmp_path / "mix.json"
    mixp.write_text(json.dumps(mix_with_exotic))
    ticket = tmp_path / "ticket.csv"
    rc = main(["predict", "--mix", str(mixp), "--ticket", str(ticket)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    lines = ticket.read_text().splitlines()
    total_line = next(line for line in lines if line.startswith("carbon_kgCO2,TOTAL,"))
    assert float(total_line.split(",")[2]) == pytest.approx(out["carbon"], abs=0.05)


def test_waste_factor_default_is_zero():
    assert DEFAULT_RUN_CONFIG["waste_factor"] == 0.0


def test_validate_waste_factor_accepts_boundary_and_typical_values():
    assert validate_waste_factor(0.0) is None
    assert validate_waste_factor(0.05) is None
    assert validate_waste_factor(0.4999) is None


@pytest.mark.parametrize("bad", [-0.1, 0.6, 0.5, "high", True])
def test_validate_waste_factor_rejects_out_of_range_and_non_numeric(bad):
    assert validate_waste_factor(bad) is not None


def test_cli_rejects_bad_waste_factor_exits_1_no_traceback(tmp_path, capsys):
    mixp = tmp_path / "mix.json"
    mixp.write_text(json.dumps(MIX))
    for bad in (-0.1, 0.6):
        cfg = tmp_path / "cfg.json"
        cfg.write_text(json.dumps({"config": {"waste_factor": bad}}))
        rc = main(["predict", "--mix", str(mixp), "--config", str(cfg)])
        assert rc == 1, bad
        err = capsys.readouterr().err
        assert "waste_factor" in err
        assert "Traceback" not in err


def test_cli_waste_factor_applies_to_metrics_and_ticket(tmp_path, capsys):
    mixp = tmp_path / "mix.json"
    mixp.write_text(json.dumps(MIX))
    ticket = tmp_path / "ticket.csv"
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"config": {"waste_factor": 0.05}}))
    rc = main(["predict", "--mix", str(mixp), "--config", str(cfg), "--ticket", str(ticket)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["carbon_as_placed"] == pytest.approx(out["carbon"] * 1.05, rel=1e-6)
    lines = ticket.read_text().splitlines()
    assert any(line == "config,waste_factor,0.05" for line in lines)
    total_line = next(line for line in lines if line.startswith("carbon_kgCO2,TOTAL,"))
    placed_line = next(line for line in lines if line.startswith("carbon_kgCO2,TOTAL_as_placed,"))
    total = float(total_line.split(",")[2])
    placed = float(placed_line.split(",")[2])
    assert placed == pytest.approx(total * 1.05, abs=0.01)


def test_cli_waste_factor_at_default_is_bit_identical(tmp_path, capsys):
    """wf=0 (the default) must not move any existing predict output field."""
    mixp = tmp_path / "mix.json"
    mixp.write_text(json.dumps(MIX))
    rc = main(["predict", "--mix", str(mixp)])
    assert rc == 0
    base = json.loads(capsys.readouterr().out)

    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"config": {"waste_factor": 0.0}}))
    rc = main(["predict", "--mix", str(mixp), "--config", str(cfg)])
    assert rc == 0
    explicit = json.loads(capsys.readouterr().out)
    assert explicit["carbon"] == base["carbon"]
    assert explicit["carbon_as_placed"] == base["carbon"]


# --- R8.0 WP-E: disclosure rows reach the CLI ticket automatically ----------------
# No new run-config keys (site temp / transport detail are UI-session concerns per
# the WP-E prompt) -- the CLI's `predict --ticket` output must still gain every new
# disclosure row through `mix_ticket`, at the module's own defaults (site_temp_c
# 20.0, transport_detail off), without any src/cli.py wiring for them.

def test_cli_predict_output_carries_disclosure_fields(tmp_path, capsys):
    mixp = tmp_path / "mix.json"
    mixp.write_text(json.dumps(MIX))
    rc = main(["predict", "--mix", str(mixp)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["carbon_interval_lo"] <= out["carbon"] <= out["carbon_interval_hi"]
    assert out["delta_t_adiabatic_C"] is not None
    assert out["curing_maturity_days"] is not None
    assert out["curing_maturity_days"] != out["curing"]  # secondary, not a switchover


def test_cli_ticket_carries_disclosure_rows(tmp_path, capsys):
    mixp = tmp_path / "mix.json"
    mixp.write_text(json.dumps(MIX))
    ticket = tmp_path / "ticket.csv"
    rc = main(["predict", "--mix", str(mixp), "--ticket", str(ticket)])
    assert rc == 0
    lines = ticket.read_text().splitlines()
    assert any(line.startswith("carbon_kgCO2,interval_lo,") for line in lines)
    assert any(line.startswith("carbon_kgCO2,interval_hi,") for line in lines)
    assert any(line.startswith("carbon_kgCO2,carbonation_uptake_bound_informational,") for line in lines)
    assert any(line.startswith("thermal,delta_t_adiabatic_C,") for line in lines)
    assert any(line.startswith("prediction,curing_maturity_days_uncalibrated,") for line in lines)
    assert any(line.startswith("allocation,cement,") for line in lines)
    # No new run-config keys: no per-material transport disclosure by default.
    assert not any(line.startswith("transport_detail,") for line in lines)


def test_cli_design_ticket_also_carries_disclosure_rows(tmp_path, capsys):
    """recommend_recipe's own dict predates WP-E; the CLI ticket must still get
    every disclosure row via mix_ticket's fallback (not just `predict`'s)."""
    ticket = tmp_path / "ticket.csv"
    rc = main(["design", "--target", "45", "--backend", "ga", "--age", "28",
               "--ticket", str(ticket)])
    assert rc == 0
    lines = ticket.read_text().splitlines()
    assert any(line.startswith("carbon_kgCO2,interval_lo,") for line in lines)
    assert any(line.startswith("thermal,delta_t_adiabatic_C,") for line in lines)
