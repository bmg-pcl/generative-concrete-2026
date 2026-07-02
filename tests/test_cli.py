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

from src.cli import main, load_mix, load_project_config, CliError
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


def test_design_tracks_target(capsys):
    rc = main(["design", "--target", "45", "--backend", "ga", "--age", "28"])
    assert rc == 0
    rec = json.loads(capsys.readouterr().out)
    assert abs(rec["strength"] - 45.0) < 6.0
    assert rec["params"]["age"] == pytest.approx(28.0, abs=1.0)
    assert set(PARAM_NAMES) <= set(rec["params"])


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
