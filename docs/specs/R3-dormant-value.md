# R3 — Dormant & Brief-Promised Value

**Status:** Draft · **Effort:** ~1–2 days · **Depends on:** none hard; R3.1 benefits from
R1's calibrated intervals

## Motivation

Features the project brief promised (CLAUDE.md) or that already exist in code but never
reach the user — plus one regression to fix:

1. `BayesFlowExplorer.suggest_tests` (active experimental design) is implemented and
   **surfaced nowhere**.
2. **Tensile strength** was in the brief and never delivered; curing time lost its
   "(heuristic)" label in a Phase C caption edit.
3. There is **no mix-ticket export** — the actual deliverable of a session.
4. **Session export is incomplete** (regression introduced in the carbon-config change):
   `get_state_json` omits `carbon_factors`, transport distance, cement type, chemistry
   mode, and the exotic-strength toggle, so a restored "complete session" silently loses
   the carbon configuration.

## 1. Active-learning panel (surface `suggest_tests`)

### Design
- Calibration tab gains a **"Suggest next experiments"** section: optional target
  strength input → `suggest_tests(target, n_tests=5)` → table of 5 mixes with predicted
  strength, 90% interval, novelty, carbon, cost, and a CSV download.
- Update the merit score to use the new machinery:
  `merit = interval_halfwidth · min(novelty, 3) / (1 + |strength − target|)` — prefer
  informative (wide-interval), under-sampled (high-novelty, capped so absurd corners
  don't dominate), on-target mixes. Keep candidates within the volume balance once R2
  lands (guard with `hasattr` until then).
- **Metrics history:** `StrengthPredictor.train()` appends
  `{timestamp, n_rows, rmse, r2}` to `models/metrics_history.json`; the Calibration tab
  renders the history as a table + line chart, so uploading lab data shows measurable
  before/after improvement.

### Acceptance gates
- Unit test: `suggest_tests` returns `n_tests` rows sorted by merit, all columns finite.
- Test: `train()` appends exactly one history record per call.
- AppTest: panel renders without exception.

## 2. Tensile strength (derived) + curing label fix

### Design
- `src/ui_logic.py::tensile_estimate(fc: float) -> float` implementing EC2:
  `f_ctm = 0.30 · fc^(2/3)` for fc ≤ 50 MPa, else `2.12 · ln(1 + (fc + 8)/10)`.
- Shown on Compare cards and the mix ticket as "Tensile (EC2 correlation, derived)" —
  it is a correlation from compressive strength, not an independent prediction, and the
  label must say so.
- Restore "(heuristic)" on the curing caption (`app.py` Compare cards).

### Acceptance gates
- Unit tests: both EC2 branches, continuity near 50 MPa (within 5%), monotonicity.
- Grep gate: the curing caption contains "heuristic".

## 3. Mix-ticket export

### Design
- `ui_logic.mix_ticket(mix, metrics, config) -> str` producing CSV (one file, two
  sections) containing: the 8 mix parameters; predicted strength + 90% interval +
  tensile estimate + curing (labeled); carbon **breakdown per material + transport**
  (reuse the factor table rather than a single total); cost breakdown; the active
  config (chemistry tier, cement type, transport km, factor overrides, robust mode);
  novelty/in-support; timestamp; and the standing disclaimer ("design exploration —
  validate physically before structural use").
- Download buttons on the Compare cards (per mix) and the recommended recipe.

### Acceptance gates
- Unit test: ticket parses as CSV, contains all sections, totals equal the sums of the
  breakdowns (carbon breakdown Σ == displayed carbon).
- AppTest: buttons render.

## 4. Session export/import completeness

### Design
- `get_state_json` adds: `carbon_factors`, `transport_km`, `cement_type`,
  `chemistry_mode`, `exotic_strength_enabled`, and a `version: 2` field.
- Import: restore all of the above when present; **v1 files (no version key) must still
  import** with defaults — `validate_session_state` treats the new keys as optional.
- Because transport/cement-type/chemistry-mode are Config-tab widgets, restoring them
  requires keyed widgets (`st.session_state["cfg_transport"]` etc.) — coordinate with
  R4.2's keyed-widget work; if R3 lands first, key just the Config widgets here.

### Acceptance gates
- Round-trip test: export → mutate state → import → all config values restored.
- Back-compat test: a v1-shaped dict imports without error.

## Out of scope
- PDF ticket rendering (CSV + on-screen first; PDF is cosmetic).
- Replacing the base dataset from the UI (brief's "import other datasets" beyond the
  overlay) — worth a follow-up spec if wanted.

## Risks
- EC2 correlation is for ordinary concrete; with exotics enabled the tensile label must
  inherit the "unvalidated" caveat (fibers change tensile behavior far more than the
  correlation admits).
