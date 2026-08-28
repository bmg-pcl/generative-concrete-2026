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
from .compliance import load_packs

DEFAULT_RUN_CONFIG = {"advanced": False, "transport_km": 0.0, "cement_type": "OPC",
                      "robust": True, "age": None, "clinker_source": None,
                      "waste_factor": 0.0}


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


def validate_slump_target(value) -> str | None:
    """Return None if a `--slump-target` value is usable, else an error message
    naming the bounds -- `validate_waste_factor`'s style. (0, 29]: 0 is excluded
    (not a workability target, and the corpus's own physical floor is > 0);
    29 cm is properties.py's own corpus ceiling (a value at or above it is
    outside the 0-29 cm range the slump model was ever trained on)."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return f"slump target must be numeric; got {value!r}."
    if not (0.0 < value <= 29.0):
        return f"slump target must be numeric in (0, 29] cm; got {value!r}."
    return None


def validate_waste_factor(value) -> str | None:
    """Return None if a `waste_factor` value is usable, else an error message.

    Boundary validation (spec R8.0 WP-A A2), mirroring `validate_clinker_source`'s
    style: 3-8% overbatch (batched-vs-placed) is typical, so [0, 0.5) keeps a
    fat-fingered value from silently producing a nonsense as-placed carbon figure."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return f"waste_factor must be numeric; got {value!r}."
    if not (0.0 <= value < 0.5):
        return f"waste_factor must be numeric in [0, 0.5); got {value!r}."
    return None


def parse_exposure_arg(value: str) -> tuple[str, str]:
    """Split a '--exposure <pack>:<class>' string into (pack_id, class_id).

    Boundary syntax validation only (R8.2 WP-3) -- whether the ids actually name
    a real pack/class is `validate_exposure`'s job below, mirroring
    `validate_clinker_source`'s two-step shape (parse the shape, then check the
    values against the live registry)."""
    if ":" not in value or value.count(":") != 1:
        raise CliError(f"--exposure must be '<pack>:<class>' (exactly one ':'); got '{value}'.")
    pack_id, cls = value.split(":", 1)
    if not pack_id or not cls:
        raise CliError(f"--exposure must be '<pack>:<class>'; got '{value}'.")
    return pack_id, cls


def validate_exposure(pack_id: str, cls: str) -> str | None:
    """Return None if `pack_id`/`cls` resolve to a real pack/class, else an error
    message naming the valid ids -- `validate_clinker_source`'s style, and the
    same reason: an unknown id must become a clean CliError at the boundary, not
    an uncaught KeyError from `check_compliance` deep inside `compute_metrics`.
    Never a hardcoded jurisdiction list -- `load_packs()` is the live registry
    (R8.2's honesty contract: a pack is a JSON drop-in, not a code change)."""
    packs = load_packs()
    if pack_id not in packs:
        return (f"Unknown exposure pack '{pack_id}'. Valid packs: "
                f"{', '.join(sorted(packs)) or '(none available)'}.")
    classes = packs[pack_id].get("classes", {})
    if cls not in classes:
        return (f"Unknown exposure class '{cls}' in pack '{pack_id}'. Valid "
                f"classes: {', '.join(sorted(classes))}.")
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
    if "waste_factor" in run:
        err = validate_waste_factor(run["waste_factor"])
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
            "waste_factor": float(cfg["run"]["waste_factor"]),
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


def _write_ticket(path: str, mix_dict_named: dict, metrics: dict, cfg: dict, exotic: dict = None,
                  extra_lines: list = None):
    from .ui_logic import mix_ticket
    csv = mix_ticket(mix_dict_named, metrics, _ticket_config(cfg), exotic=exotic)
    # `extra_lines` (R8.5 P4): mix_ticket (frozen) has no row for the strict/
    # allow-unknown compliance mode -- appended here, in the same
    # "section,key,value" shape as its own `config,*` rows, never reordering
    # anything mix_ticket already wrote.
    if extra_lines:
        csv += "\n" + "\n".join(extra_lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(csv)
    print(f"Ticket written to {path}", file=sys.stderr)


def _pick_compliant(checked, allow_unknown: bool):
    """R8.5 P4: apply OUR OWN strict-vs-allow-UNKNOWN acceptance criterion over
    `design_compliant()`'s `"checked"` list (every ranked candidate, best-first,
    with its real `check_compliance()` verdict). Strict: first verdict ==
    "PASS". Allow-unknown: first verdict in {"PASS", "UNKNOWN"} -- "UNKNOWN"
    already means zero FAILing rules (check_compliance's own aggregation: any
    FAIL wins over UNKNOWN), so this is exactly "every EVALUABLE rule passes,
    some rules just aren't sourced by the pack." Mirrors ui/inverse.py's
    identical helper (duplicated, not imported, so this module never pulls in
    Streamlit -- see test_cli_never_imports_streamlit)."""
    for mix, result in checked:
        if result["verdict"] == "PASS":
            return mix, result
        if allow_unknown and result["verdict"] == "UNKNOWN":
            return mix, result
    return None, None


def cmd_predict(args) -> int:
    from .models import StrengthPredictor
    from .ui_logic import compute_metrics
    cfg = load_project_config(args.config, epd_path=args.epd)
    mix, exotic = load_mix(args.mix)
    exposure_pack = exposure_class = None
    if args.exposure:
        exposure_pack, exposure_class = parse_exposure_arg(args.exposure)
        err = validate_exposure(exposure_pack, exposure_class)
        if err:
            raise CliError(err)
    predictor = StrengthPredictor()
    metrics = compute_metrics(mix, exotic, cfg["costs"], predictor,
                              advanced=bool(cfg["run"]["advanced"]),
                              exotic_strength=False,
                              carbon_kwargs=_carbon_kwargs(cfg),
                              waste_factor=float(cfg["run"]["waste_factor"]),
                              exposure_pack=exposure_pack, exposure_class=exposure_class)
    out = {"mix": dict(zip(PARAM_NAMES, mix)), **metrics}
    print(json.dumps(_jsonable(out), indent=2))
    if args.ticket:
        _write_ticket(args.ticket, dict(zip(PARAM_NAMES, mix)), metrics, cfg, exotic=exotic)
    return 0


def cmd_design(args) -> int:
    from .bayesian import BayesFlowExplorer
    from .ui_logic import recommend_recipe, compute_metrics, carbon_term
    from .generative_ga import SLUMP_SP_DOSING_NOTE

    cfg = load_project_config(args.config, epd_path=args.epd)
    age = cfg["run"]["age"] if args.age is None else args.age
    age = float(age) if age is not None else None

    slump_target = None
    if args.slump_target is not None:
        err = validate_slump_target(args.slump_target)
        if err:
            raise CliError(err)
        slump_target = float(args.slump_target)

    exposure_pack = exposure_class = None
    if args.exposure:
        exposure_pack, exposure_class = parse_exposure_arg(args.exposure)
        err = validate_exposure(exposure_pack, exposure_class)
        if err:
            raise CliError(err)

    explorer = BayesFlowExplorer()

    if exposure_pack is not None:
        # R8.5 P4: `recommend_recipe` (frozen, WP-1) has no `compliance` kwarg.
        # Go straight to `design_compliant()` on the metaheuristic designer (the
        # engine's own honest-verification wrapper), same approach as
        # ui/inverse.py -- GA/ACO only (`--backend flow`/`auto` have no
        # compliance-aware search; fall back to GA, same graceful downgrade the
        # UI performs, disclosed on stderr not silently).
        compliant_backend = "aco" if args.backend == "aco" else "ga"
        if args.backend not in ("ga", "aco"):
            print(f"Note: --exposure requires the GA/ACO backend; using "
                 f"'{compliant_backend}' instead of '{args.backend}'.", file=sys.stderr)
        designer = explorer.aco_designer if compliant_backend == "aco" else explorer.designer
        result = designer.design_compliant(
            float(args.target), compliance=(exposure_pack, exposure_class),
            robust=bool(cfg["run"]["robust"]), age=age, slump_target=slump_target,
        )
        mix_d, _check = _pick_compliant(result["checked"], args.allow_unknown)
        compliance_mode = "allow_unknown" if args.allow_unknown else "strict"
        if mix_d is None:
            out = {
                "found": False, "mix": None, "params": None,
                "pack_id": exposure_pack, "class": exposure_class,
                "compliance_mode": compliance_mode,
            }
            print(json.dumps(_jsonable(out), indent=2))
            if args.ticket:
                print(f"No ticket written: no design {'whose evaluable rules all pass' if args.allow_unknown else 'that verifies PASS'} "
                     f"against {exposure_pack}:{exposure_class} was found.", file=sys.stderr)
            return 0
        mix_arr = np.array([mix_d[p] for p in PARAM_NAMES])
        m = compute_metrics(mix_arr, {}, cfg["costs"], explorer.predictor,
                            advanced=bool(cfg["run"]["advanced"]),
                            carbon_kwargs=_carbon_kwargs(cfg),
                            exposure_pack=exposure_pack, exposure_class=exposure_class)
        carbon_disp = carbon_term(mix_d, bool(cfg["run"]["advanced"]), _carbon_kwargs(cfg),
                                  robust_carbon=args.robust_carbon)
        rec = {
            "mix": mix_arr, "params": mix_d, "strength": m["strength"],
            "interval_lo": m["interval_lo"], "interval_hi": m["interval_hi"],
            "novelty": m["novelty"], "in_support": m["in_support"],
            "workability": m["workability"], "tensile": m["tensile"], "curing": m["curing"],
            "carbon": carbon_disp, "carbon_basis": "upper_95" if args.robust_carbon else "point",
            "cost": m["cost"], "delta_t_adiabatic_C": m["delta_t_adiabatic_C"],
            "mass_pour_flag": m["mass_pour_flag"], "compliance": m["compliance"],
            "compliance_mode": compliance_mode,
        }
        if slump_target is not None:
            rec["slump_target"] = slump_target
            rec["slump_cm"] = m["slump_cm"]
            rec["slump_lo"] = m["slump_lo"]
            rec["slump_hi"] = m["slump_hi"]
            rec["slump_basis"] = m["slump_basis"]
            rec["slump_in_support"] = m["slump_in_support"]
            rec["slump_reason"] = m["slump_reason"]
            rec["found"] = bool(m["slump_in_support"])
            if rec["found"]:
                rec["slump_note"] = SLUMP_SP_DOSING_NOTE
        out = {k: v for k, v in rec.items() if k != "mix"}
        print(json.dumps(_jsonable(out), indent=2))
        if args.ticket:
            _write_ticket(args.ticket, rec["params"], rec, cfg,
                          extra_lines=[f"config,compliance_mode,{compliance_mode}"])
        return 0

    try:
        rec = recommend_recipe(
            explorer, float(args.target), method=args.backend,
            advanced=bool(cfg["run"]["advanced"]), costs=cfg["costs"],
            carbon_kwargs=_carbon_kwargs(cfg),
            robust=bool(cfg["run"]["robust"]),
            age=age,
            robust_carbon=args.robust_carbon,
            slump_target=slump_target,
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
    p.add_argument("--exposure", default=None,
                   help="Exposure class to check, '<pack>:<class>' (e.g. 'en206:XC4'). "
                        "Advisory only, never a certification -- see docs/specs/R8.2.")
    p.add_argument("--ticket", default=None, help="Also write a mix-ticket CSV here.")
    p.set_defaults(fn=cmd_predict)

    d = sub.add_parser("design", help="Recommend a recipe for a target strength.")
    d.add_argument("--target", type=float, required=True, help="Target strength (MPa).")
    d.add_argument("--backend", default="ga", choices=["auto", "flow", "ga", "aco"])
    d.add_argument("--age", type=float, default=None, help="Fixed design age (days); overrides config.")
    d.add_argument("--config", default=None)
    d.add_argument("--epd", default=None, help="Supplier EPD JSON (see predict --epd).")
    d.add_argument("--ticket", default=None)
    d.add_argument("--robust-carbon", action="store_true",
                   help="R8.5 P2: report the +1.96σ upper bound of carbon instead of the "
                        "point total -- a disclosure/selection-figure swap, symmetric with "
                        "robust strength. Default off (point total).")
    d.add_argument("--slump-target", type=float, default=None, metavar="CM",
                   help="R8.5 P3: bias the search toward this slump target (cm), honestly "
                        "gated by the slump model's OWN support envelope. Numeric in (0, 29]. "
                        "Default: no target (ambient).")
    d.add_argument("--exposure", default=None,
                   help="R8.5 P4: require compliance with this exposure class, "
                        "'<pack>:<class>' (e.g. 'en206:XC4') -- verified with the real "
                        "check_compliance() engine, GA/ACO backend only. Advisory only, "
                        "never a certification -- see docs/specs/R8.2.")
    d.add_argument("--allow-unknown", action="store_true",
                   help="With --exposure: accept a design whose EVALUABLE rules all PASS "
                        "even when some rules are UNKNOWN (unsourced by the pack). Default: "
                        "strict -- UNKNOWN counts as a violation.")
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
