# UI Fix Plan: Making the Streamlit App Coherent

*Companion to `docs/FIX_PLAN.md`, covering `app.py`. Findings came from a review of
the running UI; the goal is a coherent app where every number matches the model/tier
that produced it, nothing misleads, nothing crashes, and the generative engine we
built is actually usable from the screen.*

**Sequencing principle:** truth first (stop misleading), then stop crashes/corruption,
then performance, then surface value, then polish. Pure number-crunching is extracted
into a testable module first so the rest can be verified without a live browser.

**Critical path:** U5 (extract testable logic) → U0 (coherence) → U2 (perf) → U3 (value).
U1 (stability) is independent. Each phase is one shippable commit on the PR branch.

## Status (2026-07-01)

| Phase | State |
|---|---|
| U5 Extract testable logic (`src/ui_logic.py` + tests) | ✅ done |
| U0 Coherence & honesty (one carbon/metrics/fitness path, tab-2 reconcile, refs, naming, exotics switch) | ✅ done |
| U1 Stability (CSV + session validation, KDE guard) | ✅ done |
| U2 Performance (cached tab-2 sampling, batched predicts) | ✅ done |
| U3 Surface value (recipe recommender + load buttons, uncertainty on cards) | ✅ done |
| U4 Polish (`title_font`, dark theme → `.streamlit/config.toml`) | ✅ done |

All verified via `src/ui_logic.py` unit tests (Streamlit-independent) + an `import app`
smoke test; full suite 34 passed.

---

## U5 — Extract testable logic (do first; unblocks verification)
Move the metric/fitness/recipe math out of `app.py` into `src/ui_logic.py`
(`compute_metrics`, `carbon_for_mode`, `scalarized_fitness`, `recommend_recipe`), so
the Streamlit script only wires widgets to functions. Add unit tests. This lets every
later coherence fix be verified with `pytest`, not a browser.

**Gate:** logic has tests; `app.py` calls the shared functions; suite green.

---

## U0 — Coherence & honesty (what users read and believe)
1. **One carbon path, mode-aware everywhere (#6).** `carbon_for_mode(mix, advanced)`
   used by tab 1 cards, tab 2 hover, tab 3 fitness AND the plotted Pareto points — so
   the "Chemistry Mode" toggle changes every carbon number consistently (today tab 1
   and the Pareto axis ignore it).
2. **Reconcile Tab 2 with reality + expose backends (#3).** Replace the stale
   banner/footnote (they claim "GA / future work" and separately "trained neural flow"
   while the code auto-uses the real flow). Add a backend selector
   `auto / flow / ga / aco`, and show which backend actually ran and whether trained
   flow weights were found.
3. **Fix References version table (#9).** Match `requirements.txt`
   (TensorFlow 2.14, TFP 0.22, XGBoost 3.2, BayesFlow 1.x).
4. **Naming consistency.** Align tab label ↔ header; soften preset labels that assert
   unverified properties ("High Strength (Row 42)").
5. ✅ **Exotics strength switch (#1)** — *done* (`cba60d3`): `src/exotics.py` +
   sidebar toggle, off by default, honest per-card captions.

**Gate:** every strength/carbon number shown matches the model/tier that produced it;
no tab-2 text claims something the code isn't doing.

---

## U1 — Stability (don't crash, don't corrupt)
6. **Calibration CSV validation (#4).** Verify the 9 required columns exist and are
   numeric before appending/retraining; back up the overlay first; clear error on bad
   input.
7. **KDE singular-axis guard (#10).** Wrap `gaussian_kde`; on a near-constant axis fall
   back to a 2D scatter instead of crashing the tab.
8. **Session-import validation (#12).** Validate keys/shapes on JSON import; friendly
   error instead of a `KeyError` traceback.

**Gate:** a malformed CSV, a degenerate target, and a bad session file each produce a
message, not a stack trace or a corrupted overlay.

---

## U2 — Performance (depends on U0 tab-2 rework)
9. **Gate expensive Tab-2 compute (#2).** `@st.cache_data` keyed on
   `(target, backend, chem_mode)` (and/or an explicit "Explore" button) so unrelated
   reruns (a tab-1 slider) don't retrigger a 3000-sample search + KDE.
10. **Batch predictions (#5).** Replace the 300-call hover loop and per-mix `predict`
    with `predict_batch`; vectorize hover carbon/cost.

**Gate:** moving a tab-1 slider no longer runs a tab-2 search; tab-2 renders in well
under a second with flow weights present.

---

## U3 — Surface the value (depends on U0 backends)
11. **"Recommend a recipe" (#8).** In tab 2, return the single best mix
    (`recommend_recipe` → `best_mix`) for the target via the selected backend, with its
    predicted strength/carbon/cost and a "load into Mix A / Mix B" button.
12. **Uncertainty: show it (#7).** Surface `evaluate_uncertainty` as a small badge on
    the Mix A/B cards (it's deterministic now) instead of computing it and discarding it.

**Gate:** a user can go from "I need 45 MPa" to a concrete, in-envelope recipe they can
load and compare, with an uncertainty indicator.

---

## U4 — Polish & future-proofing (low risk, last)
13. **De-deprecate** Plotly/Streamlit calls (`titlefont`→`title_font`; resolve the
    `use_container_width` warning).
14. **CSS robustness (optional).** Move dark-mode styling from injected `data-baseweb`
    selectors to `.streamlit/config.toml` so upgrades don't silently break it.

---

### Dependency summary
```
U5 (extract+test) ─► U0 (coherence) ─┬─► U2 (perf) ─► U3 (value)
                                     └─► U4 (polish)
U1 (stability) ── independent
```
