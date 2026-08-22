"""R8.2 WP-2 — jurisdiction exposure packs.

Structural gates (JSON validity, schema shape, sourcing-honesty markers,
cross-pack divergence) run unconditionally: they do not need the WP-1 engine.
Engine-backed gates (`validate_pack`) run only if `src.compliance` is
importable and exposes `validate_pack` -- if WP-1 hasn't landed yet, those
tests report SKIPPED with a clear reason rather than failing or silently
passing. See docs/specs/R8.2-exposure-compliance.md.
"""
import json
import os

import pytest

PACKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "exposure_packs")

PACK_FILES = {
    "en206": "en206.json",
    "bs8500": "bs8500.json",
    "aci318": "aci318.json",
}

EN_CLASS_FAMILIES = ("XC", "XD", "XS", "XF", "XA")
RULE_KEYS = ("max_w_b", "min_cement_kg_m3", "min_strength_MPa", "min_air_pct")


def _load(name):
    with open(os.path.join(PACKS_DIR, PACK_FILES[name]), encoding="utf-8") as f:
        return json.load(f)


def _try_import_validate_pack():
    """Return validate_pack if src.compliance is importable and exposes it,
    else None. WP-1 (the engine) is a sibling package that may not have
    landed yet -- this package must stay committable and green on its own."""
    try:
        from src.compliance import validate_pack
    except ImportError:
        return None
    return validate_pack


# --- unconditional structural gates -------------------------------------------------

@pytest.mark.parametrize("name", list(PACK_FILES))
def test_json_parses(name):
    pack = _load(name)
    assert isinstance(pack, dict)


@pytest.mark.parametrize("name", list(PACK_FILES))
def test_required_top_level_fields(name):
    pack = _load(name)
    for field in ("pack_id", "name", "jurisdiction", "source", "strength_basis", "classes"):
        assert field in pack, f"{name}: missing top-level field '{field}'"
    assert pack["pack_id"] == name
    assert pack["strength_basis"] in ("conformal_lower_bound", "point")
    assert pack["classes"], f"{name}: no classes defined"


@pytest.mark.parametrize("name", list(PACK_FILES))
def test_source_block_and_verified_false(name):
    """Every pack carries source.verified: false plus a verification_note --
    the schema's sourcing-honesty contract (R8.2 'honesty problem'). A `true`
    would need human sign-off and a deliberate test change; that friction is
    the point."""
    pack = _load(name)
    source = pack["source"]
    for field in ("standard", "table", "verified", "verification_note"):
        assert field in source, f"{name}: source missing '{field}'"
    assert source["verified"] is False, f"{name}: source.verified must be false pending human sign-off"
    assert isinstance(source["verification_note"], str) and len(source["verification_note"]) > 20
    assert isinstance(source["standard"], str) and len(source["standard"]) > 0
    assert isinstance(source["table"], str) and len(source["table"]) > 0


@pytest.mark.parametrize("name", list(PACK_FILES))
def test_every_limit_is_numeric_or_null(name):
    """max_w_b, min_cement_kg_m3, min_strength_MPa, min_air_pct must each be
    a number or null when present (absent means UNKNOWN and is also fine)."""
    pack = _load(name)
    for class_id, rec in pack["classes"].items():
        for key in RULE_KEYS:
            if key not in rec:
                continue  # absent is legitimate -> engine reports UNKNOWN
            val = rec[key]
            assert val is None or isinstance(val, (int, float)), (
                f"{name}.{class_id}.{key} must be numeric or null, got {val!r}"
            )
        if "max_scm_fraction" in rec and rec["max_scm_fraction"] is not None:
            frac = rec["max_scm_fraction"]
            assert isinstance(frac, dict)
            for k, v in frac.items():
                assert isinstance(v, (int, float)) and 0.0 <= v <= 1.0, (
                    f"{name}.{class_id}.max_scm_fraction[{k}] out of [0,1]: {v!r}"
                )


@pytest.mark.parametrize("name", list(PACK_FILES))
def test_no_class_with_every_rule_absent(name):
    """A class must expose at least one of the four rule keys (numeric or
    null) -- a class with every key entirely absent is not a class, it is a
    placeholder, which the spec forbids."""
    pack = _load(name)
    for class_id, rec in pack["classes"].items():
        present = [k for k in RULE_KEYS if k in rec]
        assert present, f"{name}.{class_id}: every rule key is absent"


def test_aci_taxonomy_is_not_the_en_set():
    """ACI 318 uses exposure categories F/S/W/C -- a genuinely different
    taxonomy, not a relabelling of the EN XC/XD/XS/XF/XA classes."""
    aci = _load("aci318")
    en = _load("en206")
    aci_ids = set(aci["classes"])
    en_ids = set(en["classes"])
    assert aci_ids.isdisjoint(en_ids), f"ACI class ids overlap the EN set: {aci_ids & en_ids}"
    # And the ACI ids should look like the F/S/W/C categories, not XC/XD/XS/XF/XA.
    for class_id in aci_ids:
        assert not class_id.startswith(EN_CLASS_FAMILIES), (
            f"aci318 class '{class_id}' looks like a relabelled EN class"
        )
        assert class_id[0] in "FSWC", f"unexpected ACI category letter in '{class_id}'"


def test_bs8500_diverges_from_en206_on_a_shared_class():
    """The whole point of shipping a national annex: at least one class the
    two packs share must carry a genuinely different limit. Identical packs
    would mean one was copied rather than sourced (spec, WP-2 gates)."""
    en = _load("en206")
    bs = _load("bs8500")
    shared = set(en["classes"]) & set(bs["classes"])
    assert shared, "en206 and bs8500 share no class ids to compare"

    differences = []
    for class_id in sorted(shared):
        en_rec, bs_rec = en["classes"][class_id], bs["classes"][class_id]
        for key in RULE_KEYS:
            if key in en_rec and key in bs_rec and en_rec[key] != bs_rec[key]:
                differences.append((class_id, key, en_rec[key], bs_rec[key]))
    assert differences, "bs8500 is identical to en206 on every shared class/rule -- must genuinely differ"


def test_packs_are_not_wholesale_identical():
    """Sanity check distinct from the rule-level divergence above: the two
    packs must not be byte-identical after removing pack-identity fields."""
    en = _load("en206")
    bs = _load("bs8500")
    assert en["classes"] != bs["classes"]


@pytest.mark.parametrize("name", list(PACK_FILES))
def test_pack_records_what_it_omitted(name):
    """Each pack must document, in top-level or class-level notes, what was
    left out and why -- the sourcing-honesty account the spec asks for."""
    pack = _load(name)
    top_notes = pack.get("notes", "")
    class_notes = " ".join(rec.get("notes", "") for rec in pack["classes"].values())
    combined = (top_notes + " " + class_notes).lower()
    assert "omit" in combined, f"{name}: no notes explain what was omitted and why"


# --- engine-backed gates (skipped if WP-1's src.compliance isn't present yet) -------

@pytest.mark.parametrize("name", list(PACK_FILES))
def test_pack_passes_validate_pack(name):
    validate_pack = _try_import_validate_pack()
    if validate_pack is None:
        pytest.skip("src.compliance.validate_pack not available yet (WP-1 engine not landed)")
    pack = _load(name)
    err = validate_pack(pack)
    assert err is None, f"{name} failed validate_pack: {err}"
