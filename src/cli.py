"""Headless CLI — run a particular configuration without the browser (spec R5.2).

    python -m src.cli predict --mix mix.json [--config project.json] [--ticket out.csv]
    python -m src.cli design  --target 45 [--backend ga|aco|flow|auto] [--config ...]
    python -m src.cli pareto  [--algorithm nsga2|nsga3] [--pop 60] [--gen 40] [--out front.csv]

The config file uses the SAME schema as the app's session export (gmd_session.json):
`costs` and `carbon_factors` are read if present, everything else is ignored. An
optional `config` block sets run options:

    {"costs": {...}, "carbon_factors": {...},
     "config": {"advanced": false, "transport_km": 0.0, "cement_type": "OPC",
                "robust": true, "age": 28}}

A mix file is either named params {"cement": 350, ...} (all 8 required, plus an
optional "exotic" dosing dict) or a plain 8-vector [350, 100, ...].

Imports from src/ only — no Streamlit. Exit codes: 0 ok, 1 input/validation error,
2 environment (missing optional deps / artifacts).
"""
import argparse
import json
import sys
from datetime import datetime, timezone

import numpy as np

from .generative_ga import PARAM_NAMES
from .chemistry_simple import UNIT_COSTS, CARBON_FACTORS
from .chemistry_advanced import FUEL_EF, GRID_EF
from .exotics import EXOTIC_ADMIXTURES
from .materials import validate_epd_json, carbon_provenance

DEFAULT_RUN_CONFIG = {"advanced": False, "transport_km": 0.0, "cement_type": "OPC",
                      "robust": True, "age": None, "clinker_source": None}


class CliError(Exception):
    """Input/validation error -> exit code 1 with the message on stderr."""


def validate_clinker_source(src) -> str | None:
    """Return None if a `clinker_source` descriptor is usable, else an error message.

    Mirrors `materials.validate_epd_json`'s shape: this is boundary validation of
    CLI-supplied input (spec R7.5 WP-4). `ui/config.py` cannot produce a bad key here
    since it builds its selectboxes from `sorted(FUEL_EF.keys())` and
    `sorted(GRID_EF.keys())` — but a hand-edited project config can name anything, and
    without this check an unknown fuel/grid reaches `FUEL_EF[fuel]` / `GRID_EF[elec]`
    inside `clinker_scope_split` as an uncaught KeyError instead of a clean CliError."""
    if src is None:
        return None            # the legitimate default (DEFAULT_RUN_CONFIG) — nothing to check
    if not isinstance(src, dict):
        return "clinker_source must be a JSON object."
    if "kiln_fuel" in src and src["kiln_fuel"] not in FUEL_EF:
        return (f"Unknown kiln_fuel '{src['kiln_fuel']}'. Valid fuels: "
                f"{', '.join(sorted(FUEL_EF))}.")
    if "electricity" in src and src["electricity"] not in GRID_EF:
        return (f"Unknown electricity '{src['electricity']}'. Valid grids: "
                f"{', '.join(sorted(GRID_EF))}.")
    capture = src.get("capture")
    if capture is not None:
        if not isinstance(capture, dict):
            return "clinker_source.capture must be a JSON object."
        if "rate" in capture:
            rate = capture["rate"]
            if not isinstance(rate, (int, float)) or isinstance(rate, bool) \
                    or not (0.0 <= rate <= 1.0):
                return f"clinker_source.capture.rate must be numeric in [0, 1]; got {rate!r}."
        if "energy_kwh_per_tCO2" in capture:
            energy = capture["energy_kwh_per_tCO2"]
            if not isinstance(energy, (int, float)) or isinstance(energy, bool) \
                    or energy < 0:
                return ("clinker_source.capture.energy_kwh_per_tCO2 must be numeric "
                        f"and >= 0; got {energy!r}.")
    return None


def _load_json(path: str, what: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except OSError as e:
        raise CliError(f"Cannot read {what} file {path}: {e}")
    except ValueError as e:
        raise CliError(f"{what} file {path} is not valid JSON: {e}")


def load_project_config(path: str = None, epd_path: str = None) -> dict:
    """Costs, carbon factors, and run options from a session-export-schema file.

    Resolution order for the effective carbon factors (spec R6 §3): the registry
    defaults, then attached supplier EPDs (--epd), then explicit `carbon_factors`
    in the config file (a project override beats an EPD)."""
    cfg = {"costs": UNIT_COSTS.copy(), "carbon_factors": CARBON_FACTORS.copy(),
           "run": DEFAULT_RUN_CONFIG.copy(), "epds": {}}
    if epd_path is not None:
        epd_data = _load_json(epd_path, "EPD")
        err = validate_epd_json(epd_data)
        if err:
            raise CliError(err)
        cfg["epds"] = epd_data["epds"]
        for mat, rec in cfg["epds"].items():
            if mat in cfg["carbon_factors"]:
                cfg["carbon_factors"][mat] = float(rec["value"])
    if path is None:
        return cfg
    data = _load_json(path, "config")
    if not isinstance(data, dict):
        raise CliError("Config file must be a JSON object.")
    if "costs" in data:
        cfg["costs"].update(data["costs"])
    if "carbon_factors" in data:
        cfg["carbon_factors"].update(data["carbon_factors"])
    if "epds" in data and not cfg["epds"]:   # session-export files may carry EPDs
        cfg["epds"] = data["epds"]
    run = data.get("config", {})
    unknown = set(run) - set(DEFAULT_RUN_CONFIG)
    if unknown:
        raise CliError(f"Unknown config option(s): {', '.join(sorted(unknown))}.")
    if run.get("clinker_source") is not None:
        err = validate_clinker_source(run["clinker_source"])
        if err:
            raise CliError(err)
    cfg["run"].update(run)
    return cfg


def load_mix(path: str):
    """Return (mix_vector, exotic_dict) from a mix file."""
    data = _load_json(path, "mix")
    exotic = {k: 0 for k in EXOTIC_ADMIXTURES}
    if isinstance(data, list):
        if len(data) != len(PARAM_NAMES):
            raise CliError(f"Mix vector must have {len(PARAM_NAMES)} values "
                           f"({', '.join(PARAM_NAMES)}); got {len(data)}.")
        return np.asarray(data, dtype=float), exotic
    if isinstance(data, dict):
        missing = [p for p in PARAM_NAMES if p not in data]
        if missing:
            raise CliError(f"Mix file missing parameter(s): {', '.join(missing)}.")
        for k, v in data.get("exotic", {}).items():
            if k not in exotic:
                raise CliError(f"Unknown exotic admixture: {k}.")
            exotic[k] = float(v)
        return np.array([float(data[p]) for p in PARAM_NAMES]), exotic
    raise CliError("Mix file must be a JSON object of named params or an 8-vector.")


def _carbon_kwargs(cfg: dict) -> dict:
    return {"transport_km": float(cfg["run"]["transport_km"]),
            "cement_type": cfg["run"]["cement_type"],
            "factors": cfg["carbon_factors"],
            "clinker_source": cfg["run"]["clinker_source"]}


def _ticket_config(cfg: dict) -> dict:
    return {**_carbon_kwargs(cfg), "advanced": bool(cfg["run"]["advanced"]),
            "costs": cfg["costs"], "robust": bool(cfg["run"]["robust"]),
            "carbon_provenance": carbon_provenance(cfg["carbon_factors"], cfg["epds"]),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _write_ticket(path: str, mix_dict_named: dict, metrics: dict, cfg: dict):
    from .ui_logic import mix_ticket
    with open(path, "w", encoding="utf-8") as f:
        f.write(mix_ticket(mix_dict_named, metrics, _ticket_config(cfg)))
    print(f"Ticket written to {path}", file=sys.stderr)


def cmd_predict(args) -> int:
    from .models import StrengthPredictor
    from .ui_logic import compute_metrics
    cfg = load_project_config(args.config, epd_path=args.epd)
    mix, exotic = load_mix(args.mix)
    predictor = StrengthPredictor()
    metrics = compute_metrics(mix, exotic, cfg["costs"], predictor,
                              advanced=bool(cfg["run"]["advanced"]),
                              exotic_strength=False,
                              carbon_kwargs=_carbon_kwargs(cfg))
    out = {"mix": dict(zip(PARAM_NAMES, mix)), **metrics}
    print(json.dumps(_jsonable(out), indent=2))
    if args.ticket:
        _write_ticket(args.ticket, dict(zip(PARAM_NAMES, mix)), metrics, cfg)
    return 0


def cmd_design(args) -> int:
    from .bayesian import BayesFlowExplorer
    from .ui_logic import recommend_recipe
    cfg = load_project_config(args.config, epd_path=args.epd)
    age = cfg["run"]["age"] if args.age is None else args.age
    explorer = BayesFlowExplorer()
    try:
        rec = recommend_recipe(
            explorer, float(args.target), method=args.backend,
            advanced=bool(cfg["run"]["advanced"]), costs=cfg["costs"],
            carbon_kwargs=_carbon_kwargs(cfg),
            robust=bool(cfg["run"]["robust"]),
            age=float(age) if age is not None else None,
        )
    except RuntimeError as e:   # e.g. --backend flow with no trained weights
        print(f"Error: {e}", file=sys.stderr)
        return 2
    out = {k: v for k, v in rec.items() if k != "mix"}
    print(json.dumps(_jsonable(out), indent=2))
    if args.ticket:
        _write_ticket(args.ticket, rec["params"], rec, cfg)
    return 0


def cmd_pareto(args) -> int:
    from .models import StrengthPredictor
    try:
        from .nsga import run_nsga, pymoo_available
    except ImportError:
        print("Error: pymoo is not installed (pip install pymoo).", file=sys.stderr)
        return 2
    if not pymoo_available():
        print("Error: pymoo is not installed (pip install pymoo).", file=sys.stderr)
        return 2
    cfg = load_project_config(args.config)
    age = cfg["run"]["age"] if args.age is None else args.age
    out = run_nsga(StrengthPredictor(), advanced=bool(cfg["run"]["advanced"]),
                   costs=cfg["costs"], algorithm=args.algorithm,
                   pop_size=args.pop, n_gen=args.gen,
                   carbon_kwargs=_carbon_kwargs(cfg),
                   robust=bool(cfg["run"]["robust"]),
                   age=float(age) if age is not None else None)
    header = ",".join(list(PARAM_NAMES) + ["strength", "carbon", "cost"])
    lines = [header]
    for i in range(out["front_size"]):
        row = [f"{v:.1f}" for v in out["mixes"][i]]
        row += [f"{out['strength'][i]:.1f}", f"{out['carbon'][i]:.1f}", f"{out['cost'][i]:.2f}"]
        lines.append(",".join(row))
    csv = "\n".join(lines)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(csv + "\n")
        print(f"{out['algorithm']}: {out['front_size']} mixes written to {args.out}",
              file=sys.stderr)
    else:
        print(csv)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m src.cli",
                                 description="Headless mix-design runs (see docs/specs/R5-operability.md).")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict", help="Evaluate one mix under a project config.")
    p.add_argument("--mix", required=True, help="Mix JSON (named params or 8-vector).")
    p.add_argument("--config", default=None, help="Project config JSON (session-export schema).")
    p.add_argument("--epd", default=None, help='Supplier EPD JSON ({"epds": {"cement": {"value": ...}}}).')
    p.add_argument("--ticket", default=None, help="Also write a mix-ticket CSV here.")
    p.set_defaults(fn=cmd_predict)

    d = sub.add_parser("design", help="Recommend a recipe for a target strength.")
    d.add_argument("--target", type=float, required=True, help="Target strength (MPa).")
    d.add_argument("--backend", default="ga", choices=["auto", "flow", "ga", "aco"])
    d.add_argument("--age", type=float, default=None, help="Fixed design age (days); overrides config.")
    d.add_argument("--config", default=None)
    d.add_argument("--epd", default=None, help="Supplier EPD JSON (see predict --epd).")
    d.add_argument("--ticket", default=None)
    d.set_defaults(fn=cmd_design)

    n = sub.add_parser("pareto", help="Map the strength/carbon/cost Pareto front (NSGA).")
    n.add_argument("--algorithm", default="nsga2", choices=["nsga2", "nsga3"])
    n.add_argument("--pop", type=int, default=60)
    n.add_argument("--gen", type=int, default=40)
    n.add_argument("--age", type=float, default=None)
    n.add_argument("--config", default=None)
    n.add_argument("--out", default=None, help="Write the front CSV here (default stdout).")
    n.set_defaults(fn=cmd_pareto)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except CliError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
