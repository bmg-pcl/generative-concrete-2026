"""R8.2 WP-1 -- exposure-class compliance engine tests.

Uses only data/exposure_packs/_fixture.json (owned by this work package) so
these tests never depend on WP-2's real jurisdiction packs landing. The
fixture is `_`-prefixed on purpose -- it must be excluded from load_packs()'s
default listing, and these tests assert that too.
"""
import json

import pytest

from src.compliance import (
    check_compliance,
    compare_jurisdictions,
    load_packs,
    mix_binder,
    mix_w_b,
    set_packs_path,
    validate_pack,
)

FIXTURE_PACK = load_packs(include_hidden=True)["_fixture"]


def _base_mix(**overrides):
    mix = {"cement": 300.0, "slag": 0.0, "ash": 0.0, "water": 150.0,
           "superplasticizer": 5.0, "coarse_agg": 1100.0, "fine_agg": 750.0}
    mix.update(overrides)
    return mix


# -- mix_w_b / mix_binder -----------------------------------------------------
def test_mix_binder_is_cement_plus_slag_plus_ash():
    mix = _base_mix(cement=200, slag=100, ash=50)
    assert mix_binder(mix) == 350


def test_mix_w_b_matches_chemistry_simple_convention():
    mix = _base_mix(cement=200, slag=200, ash=0, water=180)
    assert mix_w_b(mix) == pytest.approx(180 / 400)


def test_mix_w_b_zero_binder_is_inf_not_a_crash():
    mix = _base_mix(cement=0, slag=0, ash=0, water=180)
    assert mix_w_b(mix) == float("inf")


# -- fixture loading / hidden-pack exclusion ----------------------------------
def test_fixture_excluded_from_default_listing():
    packs = load_packs()
    assert "_fixture" not in packs


def test_fixture_available_via_include_hidden():
    packs = load_packs(include_hidden=True)
    assert "_fixture" in packs
    assert packs["_fixture"]["source"]["verified"] is False


# -- absent rule -> UNKNOWN with a reason -------------------------------------
def test_absent_rule_is_unknown_with_reason():
    mix = _base_mix(cement=300, water=135)  # w/b = 0.45, passes A2's max_w_b
    result = check_compliance(mix, FIXTURE_PACK, "A2")
    by_rule = {r["rule"]: r for r in result["rules"]}
    for rule_name in ("min_cement_kg_m3", "min_strength_MPa", "min_air_pct",
                       "max_scm_fraction"):
        assert by_rule[rule_name]["result"] == "UNKNOWN"
        assert by_rule[rule_name]["reason"]


# -- null rule -> skipped, never appears in rules / never affects verdict ----
def test_null_rule_is_skipped_not_unknown():
    # A1's min_air_pct is null. A fully-compliant mix should PASS outright --
    # if null were treated as UNKNOWN the verdict would wrongly be UNKNOWN.
    mix = _base_mix(cement=300, ash=50, water=150)  # w/b = 150/350 ~ 0.43
    result = check_compliance(mix, FIXTURE_PACK, "A1", strength=35)
    rule_names = [r["rule"] for r in result["rules"]]
    assert "min_air_pct" not in rule_names
    assert result["verdict"] == "PASS"


# -- boundary equality is inclusive -------------------------------------------
def test_max_w_b_exactly_at_limit_passes():
    mix = _base_mix(cement=300, ash=0, water=150)  # w/b == 0.50 exactly
    result = check_compliance(mix, FIXTURE_PACK, "A1", strength=35)
    by_rule = {r["rule"]: r for r in result["rules"]}
    assert by_rule["max_w_b"]["actual"] == pytest.approx(0.50)
    assert by_rule["max_w_b"]["result"] == "PASS"


def test_min_strength_exactly_at_limit_passes():
    mix = _base_mix(cement=300, water=150)
    result = check_compliance(mix, FIXTURE_PACK, "A1", strength_lo=30.0)
    by_rule = {r["rule"]: r for r in result["rules"]}
    assert by_rule["min_strength_MPa"]["result"] == "PASS"


def test_max_w_b_just_over_limit_fails():
    mix = _base_mix(cement=300, ash=0, water=151)  # w/b > 0.50
    result = check_compliance(mix, FIXTURE_PACK, "A1", strength=35)
    by_rule = {r["rule"]: r for r in result["rules"]}
    assert by_rule["max_w_b"]["result"] == "FAIL"


# -- UNKNOWN never yields a PASS verdict --------------------------------------
def test_unknown_never_yields_pass_verdict():
    # A2 has only max_w_b sourced; a mix that satisfies it still can't PASS
    # overall because every other rule is UNKNOWN.
    mix = _base_mix(cement=300, water=100)  # w/b = 100/300 < 0.45
    result = check_compliance(mix, FIXTURE_PACK, "A2")
    assert result["verdict"] == "UNKNOWN"
    assert result["unknown_count"] > 0


def test_fail_beats_unknown_in_verdict_aggregation():
    # A2's max_w_b FAILs; everything else UNKNOWN. FAIL must win.
    mix = _base_mix(cement=300, water=200)  # w/b > 0.45 -> FAIL
    result = check_compliance(mix, FIXTURE_PACK, "A2")
    assert result["verdict"] == "FAIL"


# -- advisory is always True for an unverified pack ---------------------------
def test_advisory_true_for_unverified_pack():
    assert FIXTURE_PACK["source"]["verified"] is False
    mix = _base_mix(cement=300, water=150)
    result = check_compliance(mix, FIXTURE_PACK, "A1", strength=35)
    assert result["advisory"] is True


def test_advisory_false_when_source_verified_true(tmp_path):
    verified_pack = json.loads(json.dumps(FIXTURE_PACK))
    verified_pack["pack_id"] = "_fixture_verified"
    verified_pack["source"]["verified"] = True
    result = check_compliance(_base_mix(cement=300, water=150), verified_pack, "A1",
                               strength=35)
    assert result["advisory"] is False


# -- strength checked against strength_lo, basis disclosed --------------------
def test_strength_lo_used_over_point_strength_with_basis_disclosed():
    mix = _base_mix(cement=300, water=150)
    result = check_compliance(mix, FIXTURE_PACK, "A1", strength=100, strength_lo=25.0)
    by_rule = {r["rule"]: r for r in result["rules"]}
    rule = by_rule["min_strength_MPa"]
    assert rule["actual"] == 25.0
    assert "conformal_lower_bound" in rule["basis"]
    assert rule["result"] == "FAIL"  # 25 < 30 required, even though point strength=100 would pass


def test_strength_point_fallback_basis_disclosed():
    mix = _base_mix(cement=300, water=150)
    result = check_compliance(mix, FIXTURE_PACK, "A1", strength=35)
    by_rule = {r["rule"]: r for r in result["rules"]}
    rule = by_rule["min_strength_MPa"]
    assert rule["actual"] == 35
    assert "point estimate" in rule["basis"]
    assert rule["result"] == "PASS"


def test_no_strength_supplied_is_unknown():
    mix = _base_mix(cement=300, water=150)
    result = check_compliance(mix, FIXTURE_PACK, "A1")
    by_rule = {r["rule"]: r for r in result["rules"]}
    assert by_rule["min_strength_MPa"]["result"] == "UNKNOWN"


# -- air content: UNKNOWN unless measured air_pct supplied --------------------
def test_air_rule_unknown_without_air_pct():
    mix = _base_mix(cement=300, water=150)
    result = check_compliance(mix, FIXTURE_PACK, "A3")
    by_rule = {r["rule"]: r for r in result["rules"]}
    assert by_rule["min_air_pct"]["result"] == "UNKNOWN"


def test_air_rule_pass_and_fail_with_measured_air_pct():
    mix = _base_mix(cement=300, water=150)
    passing = check_compliance(mix, FIXTURE_PACK, "A3", air_pct=4.0)
    failing = check_compliance(mix, FIXTURE_PACK, "A3", air_pct=3.9)
    p_rule = {r["rule"]: r for r in passing["rules"]}["min_air_pct"]
    f_rule = {r["rule"]: r for r in failing["rules"]}["min_air_pct"]
    assert p_rule["result"] == "PASS"
    assert f_rule["result"] == "FAIL"


# -- max_scm_fraction: one sub-rule per material, fraction of total binder ---
def test_max_scm_fraction_per_material():
    mix = _base_mix(cement=200, ash=66, slag=0, water=133)  # ash/binder = 66/266 ~ 0.248
    result = check_compliance(mix, FIXTURE_PACK, "A1", strength=35)
    by_rule = {r["rule"]: r for r in result["rules"]}
    assert by_rule["max_scm_fraction:ash"]["result"] == "PASS"
    assert by_rule["max_scm_fraction:slag"]["result"] == "PASS"


def test_max_scm_fraction_fail_when_over_limit():
    mix = _base_mix(cement=100, ash=100, slag=0, water=100)  # ash/binder = 0.5 > 0.33
    result = check_compliance(mix, FIXTURE_PACK, "A1", strength=35)
    by_rule = {r["rule"]: r for r in result["rules"]}
    assert by_rule["max_scm_fraction:ash"]["result"] == "FAIL"


# -- validate_pack rejects each malformed case --------------------------------
def _valid_pack():
    return json.loads(json.dumps(FIXTURE_PACK))


def test_validate_pack_accepts_the_fixture():
    assert validate_pack(FIXTURE_PACK) is None


def test_validate_pack_rejects_missing_source_verified():
    pack = _valid_pack()
    del pack["source"]["verified"]
    err = validate_pack(pack)
    assert err is not None
    assert "verified" in err


def test_validate_pack_rejects_unknown_class_field():
    pack = _valid_pack()
    pack["classes"]["A1"]["totally_made_up_field"] = 42
    err = validate_pack(pack)
    assert err is not None
    assert "unknown field" in err


def test_validate_pack_rejects_non_numeric_limit():
    pack = _valid_pack()
    pack["classes"]["A1"]["max_w_b"] = "point five"
    err = validate_pack(pack)
    assert err is not None
    assert "max_w_b" in err


def test_validate_pack_rejects_scm_fraction_outside_unit_interval():
    pack = _valid_pack()
    pack["classes"]["A1"]["max_scm_fraction"]["ash"] = 1.5
    err = validate_pack(pack)
    assert err is not None
    assert "max_scm_fraction" in err


def test_validate_pack_rejects_scm_fraction_negative():
    pack = _valid_pack()
    pack["classes"]["A1"]["max_scm_fraction"]["ash"] = -0.1
    err = validate_pack(pack)
    assert err is not None
    assert "max_scm_fraction" in err


def test_validate_pack_rejects_missing_top_level_field():
    pack = _valid_pack()
    del pack["jurisdiction"]
    err = validate_pack(pack)
    assert err is not None
    assert "jurisdiction" in err


# -- set_packs_path: a pack dropped into a temp dir surfaces with no code
#    change (mirrors tests/test_materials.py::test_new_material_via_registry_json)
def test_new_pack_via_set_packs_path(tmp_path):
    pack = _valid_pack()
    pack["pack_id"] = "temp_dir_pack"
    pack["name"] = "Temp-dir drop-in pack"
    (tmp_path / "temp_dir_pack.json").write_text(json.dumps(pack), encoding="utf-8")
    try:
        set_packs_path(str(tmp_path))
        packs = load_packs()
        assert "temp_dir_pack" in packs
        assert packs["temp_dir_pack"]["name"] == "Temp-dir drop-in pack"
        result = check_compliance(_base_mix(cement=300, water=150),
                                   packs["temp_dir_pack"], "A1", strength=35)
        assert result["verdict"] == "PASS"
    finally:
        set_packs_path(None)


def test_set_packs_path_none_resets_to_default():
    try:
        set_packs_path("/nonexistent/path/for/this/test")
        assert load_packs() == {}
    finally:
        set_packs_path(None)
    # Back on the default path, the real fixture (hidden) is reachable again.
    assert "_fixture" in load_packs(include_hidden=True)


# -- compare_jurisdictions -----------------------------------------------------
def test_compare_jurisdictions_one_row_per_pack_class():
    packs = {"_fixture": FIXTURE_PACK}
    mix = _base_mix(cement=300, water=150)
    rows = compare_jurisdictions(mix, {"_fixture": "A1"}, packs=packs, strength=35)
    assert len(rows) == 1
    assert rows[0]["pack_id"] == "_fixture"
    assert rows[0]["class"] == "A1"
    assert rows[0]["verdict"] == "PASS"


def test_compare_jurisdictions_skips_unknown_pack_id():
    packs = {"_fixture": FIXTURE_PACK}
    mix = _base_mix(cement=300, water=150)
    rows = compare_jurisdictions(mix, {"_fixture": "A1", "not_a_real_pack": "X"},
                                  packs=packs, strength=35)
    assert len(rows) == 1


def test_compare_jurisdictions_shows_national_variation():
    """Same mix, two packs with different max_w_b for the 'same' class id --
    the whole point of the feature (spec Motivation)."""
    strict = json.loads(json.dumps(FIXTURE_PACK))
    strict["pack_id"] = "_fixture_strict"
    strict["classes"]["A1"]["max_w_b"] = 0.30
    packs = {"_fixture": FIXTURE_PACK, "_fixture_strict": strict}
    mix = _base_mix(cement=300, water=150)  # w/b = 0.50
    rows = compare_jurisdictions(mix, {"_fixture": "A1", "_fixture_strict": "A1"},
                                  packs=packs, strength=35)
    verdicts = {r["pack_id"]: r["verdict"] for r in rows}
    assert verdicts["_fixture"] == "PASS"
    assert verdicts["_fixture_strict"] == "FAIL"
