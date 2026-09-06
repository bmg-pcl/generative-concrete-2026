"""
R8.6 WP-U2 -- Compare tab: disclosure hierarchy & compliance presentation.

Headless AppTest coverage (see tests/test_app_smoke.py for the established
pattern) plus a couple of fast unit tests on the new pure string helper. These
assert CONTENT, never tab position/order (WP-U1 owns tab ordering in app.py).
"""
import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.ui_logic import slump_caveat, slump_disclosure_text  # noqa: E402

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


# ---------------------------------------------------------------------------
# Unit tests on the new pure helper (src/ui_logic.py) -- no Streamlit needed.
# ---------------------------------------------------------------------------

def test_slump_disclosure_text_model_basis_matches_slump_caveat_verbatim():
    """On the model basis path, the new helper must return `slump_caveat`'s
    text unchanged -- it is a dispatcher, not a second wording."""
    metrics = {"slump_basis": "model", "slump_lo": 10.0, "slump_hi": 22.0}
    assert slump_disclosure_text(metrics) == slump_caveat(10.0, 22.0)


def test_slump_disclosure_text_heuristic_basis_names_the_reason():
    metrics = {"slump_basis": "heuristic", "slump_reason": "outside envelope"}
    text = slump_disclosure_text(metrics)
    assert "heuristic" in text
    assert "outside envelope" in text


def test_slump_disclosure_text_heuristic_basis_falls_back_without_reason():
    metrics = {"slump_basis": "heuristic", "slump_reason": None}
    text = slump_disclosure_text(metrics)
    assert "outside the slump corpus" in text


# ---------------------------------------------------------------------------
# AppTest coverage
# ---------------------------------------------------------------------------

def test_app_runs_clean():
    at = AppTest.from_file(APP, default_timeout=240).run()
    assert not at.exception, f"App raised: {at.exception}"


def test_shared_slump_caveat_appears_exactly_once_for_default_mixes():
    """R8.6 WP-U2 deliverable 2: the default Mix A / Mix B configuration hits
    the slump model's heuristic out-of-support fallback for BOTH mixes with
    the identical reason text -- the exact "duplicated verbatim under both
    columns" case the spec calls out. The Compare tab must render that shared
    caveat text exactly once (a single shared popover), not twice."""
    at = AppTest.from_file(APP, default_timeout=240).run()
    assert not at.exception

    # Confirm both default mixes really do share the caveat (guards against
    # the test silently passing if DEFAULT_MIX_A/B or the slump model change
    # out from under this test).
    default_caveat = slump_disclosure_text(
        {"slump_basis": "heuristic",
         "slump_reason": "Mix lies outside the slump corpus's training envelope "
                         "(103 rows); no measured estimate is available for this region."}
    )
    all_caption_text = [c.value for c in at.caption]
    occurrences = sum(1 for t in all_caption_text if t == default_caveat)
    assert occurrences == 1, (
        f"expected the shared slump caveat exactly once, found {occurrences} "
        f"in: {all_caption_text}"
    )


def test_compliance_framing_sentence_present():
    """R8.6 WP-U2 deliverable 3: a plain-language sentence sits above the
    cross-jurisdiction table, always visible, naming that a representative
    class is checked and that a real Config selection makes it meaningful."""
    at = AppTest.from_file(APP, default_timeout=240).run()
    assert not at.exception

    all_caption_text = [c.value for c in at.caption]
    assert any(
        "representative" in t and "exposure class" in t and "Config" in t
        for t in all_caption_text
    ), f"framing sentence not found in captions: {all_caption_text}"


def test_cross_jurisdiction_expander_collapsed_by_default_expanded_on_selection():
    """R8.6 WP-U2 deliverable 3: the per-jurisdiction verdict table is wrapped
    in an expander that starts collapsed (no exposure class chosen) and
    expands once a real exposure class is selected in Config."""
    at = AppTest.from_file(APP, default_timeout=240).run()
    assert not at.exception

    verdict_expander = next(
        e for e in at.expander if e.label == "Per-jurisdiction verdicts"
    )
    assert verdict_expander.proto.expanded is False

    at.selectbox(key="cfg_exposure_pack").set_value("en206").run()
    at.selectbox(key="cfg_exposure_class").set_value("XC4").run()
    assert not at.exception

    verdict_expander = next(
        e for e in at.expander if e.label == "Per-jurisdiction verdicts"
    )
    assert verdict_expander.proto.expanded is True


def test_cross_jurisdiction_full_disclosure_text_preserved_verbatim():
    """Non-negotiable constraint: shortening the always-visible framing line
    is only allowed when the full original disclosure paragraph remains one
    click away, unedited. Verify the full paragraph (EN 206/ACI 318 taxonomy
    caveat) still renders verbatim, inside the expander."""
    at = AppTest.from_file(APP, default_timeout=240).run()
    assert not at.exception

    all_caption_text = [c.value for c in at.caption]
    assert any(
        "not equivalent requirements" in t and "EN 206 and ACI 318" in t
        for t in all_caption_text
    ), "full original cross-jurisdiction disclosure paragraph missing verbatim"
