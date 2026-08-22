"""compliance.py -- exposure-class compliance engine (spec R8.2, WP-1).

Checks a mix design against a jurisdiction's exposure-class deemed-to-satisfy
limits (max water/binder, minimum cement content, minimum strength class,
minimum air content, maximum SCM fraction of binder). Pure functions: no
Streamlit, no `src.models` import -- the caller supplies strength (and,
ideally, its conformal lower bound) so this module stays testable and free of
a UI/model import cycle.

The honesty contract (see docs/specs/R8.2-exposure-compliance.md, "The honesty
problem this spec must solve"):

  1. Every pack's `source.verified` is required by the schema; while it is
     `false` (every real pack, for now) every result carries `advisory: True`.
  2. The engine never emits a bare boolean. Every rule resolves to PASS, FAIL,
     or UNKNOWN, and the top-level verdict aggregates as: any FAIL -> FAIL;
     else any UNKNOWN -> UNKNOWN; else PASS. UNKNOWN never upgrades to PASS.
  3. `null` and absent are different for a rule key in a pack's class: `null`
     means the rule does not apply to this class (silently skipped -- it never
     appears in the result's `rules` list, so it can never mark the verdict
     UNKNOWN or FAIL); an absent key means the pack has no sourced value for
     that rule, so the engine reports UNKNOWN with a reason rather than
     guessing.
  4. Strength is checked against the conformal LOWER BOUND when the caller
     supplies `strength_lo` (the R7.1/R7.5 interval, not the point mean) --
     the same "do not certify on a point estimate" principle robust mode
     applies to optimisation. Falling back to a bare `strength` is disclosed
     via the rule's `basis` field, never silent.

Accepted exclusions (see the spec's "Accepted exclusions" section -- do not
invent a model to make these evaluable):
  - k-value / SCM-efficiency substitution (EN 206 Sec 5.2.5): water/binder here
    uses total binder (cement + slag + ash) with no efficiency factor.
  - Air content: no model predicts it. `min_air_pct` rules report UNKNOWN
    unless the caller supplies a measured `air_pct`.
  - Cover, curing regime, chloride content by test: out of scope.

Deviation from the spec's literal `check_compliance` snippet, noted honestly:
the spec's "Result shape" section writes the signature as
`check_compliance(mix, pack, *, strength=None, ...)` with no class selector,
but the example result carries `"class": "XC4"` and `compare_jurisdictions`
needs a per-pack class id (jurisdictions do not share a taxonomy -- ACI's
categories are not EN's classes). A pack alone cannot supply which of its
classes to check, so this implementation adds `cls` as the third positional
argument: `check_compliance(mix, pack, cls, *, ...)`. This is the only way the
documented result shape and `compare_jurisdictions(mix, class_map, ...)` are
reconcilable.
"""
import glob
import json
import os
from typing import Dict, List, Optional

DEFAULT_PACKS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "exposure_packs"
)
# Overridable for tests and for plugging in an alternative pack directory
# without a code change -- mirrors src.materials.set_materials_path exactly.
_packs_path = os.environ.get("EXPOSURE_PACKS_DIR", DEFAULT_PACKS_PATH)
_cache: Dict[str, Dict[str, dict]] = {}

STRENGTH_BASES = ("conformal_lower_bound", "point")
CLASS_NUMERIC_FIELDS = ("max_w_b", "min_cement_kg_m3", "min_strength_MPa", "min_air_pct")
CLASS_FIELDS = {
    "description", "max_w_b", "min_cement_kg_m3", "min_strength_MPa",
    "min_air_pct", "max_scm_fraction", "notes",
}
PACK_REQUIRED_FIELDS = ("pack_id", "name", "jurisdiction", "source", "classes")


def set_packs_path(path: Optional[str]):
    """Point the pack registry at a different directory (None resets to the
    default). Mirrors materials.set_materials_path exactly: a jurisdiction is
    a JSON drop-in, not a code change."""
    global _packs_path
    _packs_path = path or os.environ.get("EXPOSURE_PACKS_DIR", DEFAULT_PACKS_PATH)
    _cache.clear()


def _load_all(path: str) -> Dict[str, dict]:
    """Every pack in `path`, keyed by pack_id, validated, cached per path.
    Includes `_`-prefixed (hidden) packs -- load_packs() below filters them
    out of the default listing."""
    if path not in _cache:
        packs: Dict[str, dict] = {}
        for fp in sorted(glob.glob(os.path.join(path, "*.json"))):
            with open(fp, encoding="utf-8") as f:
                pack = json.load(f)
            err = validate_pack(pack)
            if err:
                raise ValueError(f"exposure pack '{os.path.basename(fp)}': {err}")
            packs[pack["pack_id"]] = pack
        _cache[path] = packs
    return _cache[path]


def load_packs(path: Optional[str] = None, *, include_hidden: bool = False) -> Dict[str, dict]:
    """The pack registry {pack_id: pack}, cached per path.

    Excludes `_`-prefixed pack ids (e.g. "_fixture") by default so a test
    fixture can never surface in the UI's jurisdiction picker. Pass
    `include_hidden=True` (this module's own tests do) to load it explicitly.
    """
    p = path or _packs_path
    all_packs = _load_all(p)
    if include_hidden:
        return dict(all_packs)
    return {k: v for k, v in all_packs.items() if not k.startswith("_")}


def validate_pack(pack: dict) -> Optional[str]:
    """Return None if a pack is usable, else a human-readable error.

    Rejects: a missing top-level field, a missing/non-boolean
    `source.verified`, an unknown `strength_basis`, a class with a field
    outside the frozen schema's set, a non-numeric (and non-null) limit, and
    a `max_scm_fraction` entry outside [0, 1].
    """
    pid = pack.get("pack_id", "<unknown>") if isinstance(pack, dict) else "<unknown>"
    if not isinstance(pack, dict):
        return "exposure pack must be a JSON object"
    for field in PACK_REQUIRED_FIELDS:
        if field not in pack:
            return f"'{pid}' is missing '{field}'"
    source = pack["source"]
    if not isinstance(source, dict) or "verified" not in source:
        return f"'{pid}' source is missing 'verified'"
    if not isinstance(source["verified"], bool):
        return f"'{pid}' source.verified must be a boolean"
    basis = pack.get("strength_basis", "conformal_lower_bound")
    if basis not in STRENGTH_BASES:
        return f"'{pid}' has unknown strength_basis '{basis}' (expected one of {STRENGTH_BASES})"
    classes = pack["classes"]
    if not isinstance(classes, dict) or not classes:
        return f"'{pid}' must declare at least one class"
    for cls_id, cls in classes.items():
        if not isinstance(cls, dict):
            return f"'{pid}' class '{cls_id}' must be an object"
        unknown = set(cls.keys()) - CLASS_FIELDS
        if unknown:
            return f"'{pid}' class '{cls_id}' has unknown field(s) {sorted(unknown)}"
        for field in CLASS_NUMERIC_FIELDS:
            if field in cls and cls[field] is not None:
                v = cls[field]
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    return f"'{pid}' class '{cls_id}' field '{field}' must be numeric or null"
        if "max_scm_fraction" in cls and cls["max_scm_fraction"] is not None:
            scm = cls["max_scm_fraction"]
            if not isinstance(scm, dict):
                return f"'{pid}' class '{cls_id}' max_scm_fraction must be a mapping or null"
            for mat, frac in scm.items():
                if isinstance(frac, bool) or not isinstance(frac, (int, float)):
                    return (f"'{pid}' class '{cls_id}' max_scm_fraction['{mat}'] "
                            f"must be numeric")
                if not (0 <= frac <= 1):
                    return (f"'{pid}' class '{cls_id}' max_scm_fraction['{mat}'] = {frac} "
                            f"is outside [0, 1]")
    return None


def mix_binder(mix: Dict[str, float]) -> float:
    """Total binder mass (kg/m3): cement + slag + ash. Consistent with
    chemistry_simple.estimate_curing_time since R7.5 -- do not re-derive."""
    return mix.get("cement", 0.0) + mix.get("slag", 0.0) + mix.get("ash", 0.0)


def mix_w_b(mix: Dict[str, float]) -> float:
    """Water/total-binder ratio. A zero-binder mix has no defined ratio;
    returns +inf so a max_w_b rule reports FAIL rather than raising or
    silently passing (the same never-silently-pass discipline as everywhere
    else in this module)."""
    binder = mix_binder(mix)
    if binder <= 0:
        return float("inf")
    return mix.get("water", 0.0) / binder


def _rule(name, required, actual, result, basis=None, reason=None) -> dict:
    return {"rule": name, "required": required, "actual": actual,
            "result": result, "basis": basis, "reason": reason}


def check_compliance(mix: Dict[str, float], pack: dict, cls: str, *,
                      strength: Optional[float] = None,
                      strength_lo: Optional[float] = None,
                      air_pct: Optional[float] = None) -> dict:
    """Check `mix` against one exposure class of one pack.

    `cls` selects the class within `pack["classes"]` (see the module
    docstring's "Deviation from the spec" note for why this argument exists).
    Strength is checked against `strength_lo` (the conformal lower bound) when
    supplied, falling back to `strength`; neither supplied -> UNKNOWN. Air
    content is checked only when the caller supplies a measured `air_pct` --
    no model predicts it (see "Accepted exclusions").
    """
    if cls not in pack.get("classes", {}):
        raise KeyError(f"'{pack.get('pack_id', '?')}' has no class '{cls}'")
    class_def = pack["classes"][cls]
    rules: List[dict] = []

    # -- max_w_b -----------------------------------------------------------
    if "max_w_b" not in class_def:
        rules.append(_rule("max_w_b", None, None, "UNKNOWN",
                            reason="rule not present in pack"))
    elif class_def["max_w_b"] is not None:
        required = class_def["max_w_b"]
        actual = mix_w_b(mix)
        rules.append(_rule("max_w_b", required, actual,
                            "PASS" if actual <= required else "FAIL"))

    # -- min_cement_kg_m3 (raw cement mass, not total binder -- k-value SCM
    #    credit toward minimum cement content is out of scope, see module
    #    docstring's "Accepted exclusions") -----------------------------------
    if "min_cement_kg_m3" not in class_def:
        rules.append(_rule("min_cement_kg_m3", None, None, "UNKNOWN",
                            reason="rule not present in pack"))
    elif class_def["min_cement_kg_m3"] is not None:
        required = class_def["min_cement_kg_m3"]
        actual = mix.get("cement", 0.0)
        rules.append(_rule("min_cement_kg_m3", required, actual,
                            "PASS" if actual >= required else "FAIL"))

    # -- min_strength_MPa ----------------------------------------------------
    if "min_strength_MPa" not in class_def:
        rules.append(_rule("min_strength_MPa", None, None, "UNKNOWN",
                            reason="rule not present in pack"))
    elif class_def["min_strength_MPa"] is not None:
        required = class_def["min_strength_MPa"]
        if strength_lo is not None:
            actual, basis = strength_lo, "conformal_lower_bound (strength_lo supplied)"
            rules.append(_rule("min_strength_MPa", required, actual,
                                "PASS" if actual >= required else "FAIL", basis=basis))
        elif strength is not None:
            actual, basis = strength, "point estimate (no strength_lo supplied)"
            rules.append(_rule("min_strength_MPa", required, actual,
                                "PASS" if actual >= required else "FAIL", basis=basis))
        else:
            rules.append(_rule("min_strength_MPa", required, None, "UNKNOWN",
                                reason="no strength or strength_lo supplied"))

    # -- min_air_pct (no air-content model; see "Accepted exclusions") -------
    if "min_air_pct" not in class_def:
        rules.append(_rule("min_air_pct", None, None, "UNKNOWN",
                            reason="rule not present in pack"))
    elif class_def["min_air_pct"] is not None:
        required = class_def["min_air_pct"]
        if air_pct is None:
            rules.append(_rule("min_air_pct", required, None, "UNKNOWN",
                                reason="no air-content model; air_pct not supplied"))
        else:
            rules.append(_rule("min_air_pct", required, air_pct,
                                "PASS" if air_pct >= required else "FAIL"))

    # -- max_scm_fraction (one sub-rule per material in the mapping) --------
    if "max_scm_fraction" not in class_def:
        rules.append(_rule("max_scm_fraction", None, None, "UNKNOWN",
                            reason="rule not present in pack"))
    elif class_def["max_scm_fraction"] is not None:
        binder = mix_binder(mix)
        for mat, limit in class_def["max_scm_fraction"].items():
            qty = mix.get(mat, 0.0)
            actual = 0.0 if binder <= 0 else qty / binder
            rules.append(_rule(f"max_scm_fraction:{mat}", limit, actual,
                                "PASS" if actual <= limit else "FAIL"))

    unknowns = [r for r in rules if r["result"] == "UNKNOWN"]
    if any(r["result"] == "FAIL" for r in rules):
        verdict = "FAIL"
    elif unknowns:
        verdict = "UNKNOWN"
    else:
        verdict = "PASS"

    source = pack.get("source", {})
    return {
        "pack_id": pack.get("pack_id"),
        "class": cls,
        "verdict": verdict,
        "advisory": not bool(source.get("verified", False)),
        "rules": rules,
        "unknown_count": len(unknowns),
        "source": source,
    }


def compare_jurisdictions(mix: Dict[str, float], class_map: Dict[str, str],
                           packs: Optional[Dict[str, dict]] = None,
                           **kw) -> List[dict]:
    """Check the same mix across jurisdictions -- the national-variation table.

    `class_map` maps pack_id -> class id within that pack (jurisdictions do
    not share a taxonomy -- e.g. ACI's F/S/W/C categories are not EN's
    XC/XD/XS/XF/XA classes -- so the class id must be given per pack). Packs
    default to `load_packs()` (the public, non-hidden set); pass `packs=` to
    supply an explicit set, e.g. the `_fixture` pack in this module's own
    tests. `**kw` (strength, strength_lo, air_pct) is forwarded to every
    `check_compliance` call. One result row per (pack, class), skipping any
    pack_id in `class_map` that is not in `packs`.
    """
    packs = packs if packs is not None else load_packs()
    rows = []
    for pack_id, cls in class_map.items():
        pack = packs.get(pack_id)
        if pack is None:
            continue
        rows.append(check_compliance(mix, pack, cls, **kw))
    return rows
