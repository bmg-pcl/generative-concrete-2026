# R7 — Trustworthy to the Edge

**Status:** R7.1 ✅ · R7.2 ✅ · R7.3 ✅ · R7.4 ✅ — all shipped. **Effort (remaining):** none.
**Depends on:** R7.1 (joint model) and R7.2 (age-conditioned flow) — both shipped.

Horizon 1's theme: the robust mode is the flagship, so its foundations must be beyond
reproach, and the amortized flow must work under the *default* configuration. R7.1
(one coherent distribution) and R7.2 (the flow conditions on (strength, age), so a
pinned age uses it) are done. R7.3 makes the *evidence* match the flagship; R7.4
clears the bounded debt the earlier phases consciously deferred.

---

## R7.3 — Evidence hardening for the flagship

### Motivation

`docs/BENCHMARKS.md` currently measures **non-robust** sampling only — i.e. the mode
users are told *not* to default to. Robust mode is ON by default, yet no committed
number characterizes it. Second, the CI coverage floor (60%) sits well under actual
coverage, so it no longer guards against regressions. Third, we started recording
actual-vs-estimate effort in R7.1/R7.2 prose but never in a durable place.

### Current state

- `scripts/benchmark_backends.py :: run_backend_benchmarks(explorer, backends,
  targets, n_samples, seed, age)` — measures MAE (`|median − target|`), cloud spread,
  in-support %, time. No `robust` axis. `run_nsga_benchmark` runs non-robust NSGA-II.
- CI: `.github/workflows/tests.yml` runs `ruff check .` then
  `pytest … --cov=src --cov-fail-under=60`. CI does **not** install TF/BayesFlow, so
  `src/amortized.py` and the flow tests are uncovered/skipped there — CI coverage is
  lower than a full local run and is the number the floor must sit below.
- Roadmap effort reconciliation lives only in `docs/specs/README.md` prose.

### Design

**R7.3a — Robust benchmark rows.** Add a `robust` parameter to
`run_backend_benchmarks`; when `robust=True`:
- **GA / ACO** honor it (their objective becomes the conformal lower bound + OOS
  penalty), so report a **lower-bound MAE** = `mean|predict_interval(sample).lo −
  target|` *in addition to* the median MAE, plus in-support %.
- **flow** does **not** recondition on robust inside `sample_posterior` (robust
  handling for the flow is a downstream filter in `recommend_recipe`), so **do not
  emit a robust flow row** — or emit it clearly labelled "sampling is unconditional;
  robust selection happens at recommend time". Prefer omitting it to avoid implying
  the flow reconditioned.
- **NSGA** run with `robust=True` (in-support constraint) — expect in-support % ≈ 100
  and report front hypervolume so the robust/non-robust trade is visible.
Render a new "Robust mode" section in `BENCHMARKS.md` beneath the existing table; the
narrative should state the trade plainly: robust buys the in-support guarantee at some
target-tightness / diversity cost.

**R7.3b — Ratchet the CI coverage floor.** **Measure first, then set.** Reproduce the
CI environment's coverage (no TF/BayesFlow) and set the floor a few points *below*
that number — never guess 68. Update the floor in `tests.yml` and the two README
lines that quote it (Testing section + the binary-artifact note).

**R7.3c — Effort log.** Add a small "Effort log" table to `docs/specs/README.md`
(spec, estimate, actual, note) and backfill R1–R7.2 from the existing prose. Cheap,
compounding: it is the only way the estimates ever get better.

### Acceptance gates
- `BENCHMARKS.md` has a Robust-mode section with lower-bound MAE + in-support % for
  GA/ACO and a robust NSGA row (in-support ≈ 100%); the benchmark smoke test still
  passes and covers the new `robust` column.
- CI green at the new floor; a deliberate 1-line coverage drop would fail it (sanity
  check locally, do not commit the drop).
- Effort-log table present and backfilled.

---

## R7.4 — Debt sweep (bounded)

### Motivation

Three deferrals from earlier phases, each small and each now cheap given the patterns
already in place. None is user-visibly broken; together they remove "the docs say X is
deferred" footnotes and make `materials.json` and the session file complete.

### Current state

- **Session restore is partial (the R3.4 deferral).** `ui/state.py` persists
  `SESSION_FIELDS = (mix_a, mix_b, costs, carbon_factors, exotic_a, exotic_b, epds)`
  (v3). The **Config** tab's own controls are *not* persisted: `chemistry_mode`,
  `transport_km`, `cement_source`, the clinker source (`fuel/electricity/capture`),
  `robust_mode`, `fix_age`, `design_age`, `exotic_strength_enabled`. These widgets are
  currently **unkeyed** (they reset to defaults every run). So exporting/importing a
  session silently loses the entire analysis configuration.
- **Core sliders are hardcoded (the R6.4 deferral).** `ui/state.py :: SLIDER_SPECS`
  is a literal list of `(param, label, min, max)`. `data/materials.json` core records
  have no `slider` block, so `materials.json` is *not* the single authority — a slider
  range change needs a code edit.
- **Binary-artifact policy is stated but not decided.** `models/` (~7 MB) +
  `docs/images/` (~3.6 MB) live in plain git; README says "revisit if it outgrows
  this" without a number.

### Design

**R7.4a — Full session restore.** Make the ten Config controls keyed widgets
(`key="cfg_chemistry_mode"`, `"cfg_transport_km"`, `"cfg_cement_source"`,
`"cfg_kiln_fuel"`, `"cfg_electricity"`, `"cfg_capture_rate"`, `"cfg_robust"`,
`"cfg_fix_age"`, `"cfg_design_age"`, `"cfg_exotic_strength"`), initialise them once in
`init_session_state()` via `setdefault`, add a nested `"ui_config"` block to
`export_session`/`apply_session`, and bump `SESSION_VERSION` to 4 (older files keep
defaults for missing fields — the existing "accept v1/v2/v3" pattern). The
value-level round-trip test (`tests/test_session_roundtrip.py`) and its key-symmetry
test extend to cover the new fields.

**Naming deviation from the original design (found during implementation):** this
block is named `"ui_config"`, not `"config"`. `src/cli.py` already reads a top-level
`"config"` key with a *different*, semantic schema (`advanced`/`transport_km`/
`cement_type`/`robust`/`age`/`clinker_source`) and raises `CliError` on unknown keys
under it. A same-named UI block full of raw `cfg_*` widget keys would collide and
break `python -m src.cli ... --config <exported-session.json>` — exactly the kind of
integration bug the R5 CLI spec's "one config dialect" design was meant to prevent.
`"ui_config"` avoids the collision with zero CLI changes; a regression test
(`tests/test_cli.py::test_config_accepts_real_ui_session_export`) feeds a real
`export_session()` output through `load_project_config` and asserts it does not
raise. If you're implementing this from the spec text alone, check for this collision
before choosing a field name — don't reuse `"config"`.

**R7.4b — Registry-driven core sliders.** Add a `slider: {min, max, label}` block to
each **trained_feature** record in `data/materials.json`; add
`materials.slider_specs_view()` returning the `SLIDER_SPECS` shape in **PARAM_NAMES
order**; `ui/state.py` reads `SLIDER_SPECS = slider_specs_view()`. The 8 model
features and their order are fixed (the model expects PARAM_NAMES order) — a test pins
the registry-derived order against `PARAM_NAMES`. Note in the record schema that the
slider range is a *display* range, distinct from the data envelope the optimizers use.

**Ordering deviation from the original design (found during implementation):**
`materials.slider_specs_view()` returns `{material: {label, min, max}}` **unordered**
(dict order = registry file order, `age` absent), not a `PARAM_NAMES`-ordered list.
`src/materials.py` sits *below* `src/generative_ga.py` in the import graph
(`chemistry_simple.py`/`physical.py`/`exotics.py` already import FROM `materials.py`,
and `generative_ga.py` imports `chemistry_simple.py`) — importing `PARAM_NAMES` into
`materials.py` to order the output would be circular. `PARAM_NAMES` is also 8 names
but the registry only has 7 core materials (`age` is a design condition, not a
material — no density/cost/carbon, no registry record). `ui/state.py` — which can
safely import both `src.generative_ga.PARAM_NAMES` and
`src.materials.slider_specs_view` — does the ordering: it iterates `PARAM_NAMES`,
looks each material up in `slider_specs_view()`, and special-cases `age` with a
hardcoded `("age", "Age (days)", 1, 365)` tuple. The acceptance gate (`[p for p,*_ in
SLIDER_SPECS] == PARAM_NAMES`) is unchanged; only the layer that produces the
ordering moved from `materials.py` to `ui/state.py`.

**R7.4c — git-lfs decision.** Measure (`git count-objects -vH`; `du -sh models
docs/images`). If the packed repo is under a stated ceiling (proposal: 50 MB), record
"plain git; revisit at 50 MB" in the README artifact note with the current number;
otherwise migrate `models/` + `docs/images/` to git-lfs and document the `.gitattributes`.
Decision + one doc paragraph; no code beyond possibly `.gitattributes`.

**Measured 2026-07-10:** `git count-objects -vH` → 13.4 MiB total; `models/` 6.8 MiB,
`docs/images/` 3.6 MiB; largest single tracked file `models/strength_quantiles.json`
at 5.3 MiB. Well under the proposed 50 MB ceiling (and GitHub's own 50/100 MB
per-file guidance). **Decision: stay on plain git — no migration.** Documented in
`README.md` under "Committed binary artifacts" with the measured numbers and a
revisit threshold (~50 MiB for a single artifact, ~250 MiB for the repo). No
`.gitattributes` change.

### Acceptance gates
- Round-trip test: exporting after changing a carbon factor **and** a Config control
  (e.g. transport_km, robust toggle) then importing into a fresh session reproduces
  both; key-symmetry test covers the new `ui_config` block.
- CLI compatibility: a real `export_session()` output, fed through
  `load_project_config`, does not raise (guards the `"config"`/`"ui_config"`
  naming-collision finding above).
- Registry gate: editing a `slider.max` in `materials.json` changes the corresponding
  Compare slider with **no code change** (test via `set_materials_path` on a temp
  JSON, as `tests/test_materials.py` already does); `slider_specs_view()` order equals
  `PARAM_NAMES`; existing bit-identical core views unchanged.
- Artifact note carries a concrete size number and a decision.
- Full suite green; AppTest smoke still exercises every tab.

---

## Implementation notes (sonnet-ready)

Read these before starting — they are where the time goes.

### The keyed-widget trap (R7.4a) — reread R4/R5 first
This is the same `StreamlitAPIException` trap R4.1/R3.4/R5.1 hit. Rules:
1. Every persisted Config control becomes a **keyed** widget; **do not** also pass a
   positional/`value=` default that fights the key. For `st.radio`/`st.selectbox`,
   the key stores the *selected value*; initialise it with `setdefault(key, default)`
   and pass `key=` only.
2. Programmatic writes (session import) must land **before** the widgets instantiate.
   `apply_session` runs in the **sidebar**, which renders before the Config tab, so
   writing the keys there is safe — mirror `load_mix_into`'s position exactly. Do
   **not** `.run()` then set keys.
3. `design_age` is conditional (only rendered when `fix_age` is on). Persist both
   `cfg_fix_age` and `cfg_design_age`; on restore, set both keys — the widget simply
   isn't drawn when `fix_age` is off, which is fine.
4. `capture_rate`/`electricity`/`kiln_fuel` are nested inside the "Clinker source"
   expander and only rendered when a non-default fuel is chosen. Persist the keys
   regardless; they're read only when that branch renders.
5. Bump `SESSION_VERSION` to 4 and keep `apply_session` tolerant: `if "ui_config" in
   data:` then set each present key — missing keys keep their `setdefault` default, so
   a v1–v3 file still imports. Add the new keys to the `ui_config` sub-dict in
   `export_session`, and to the symmetry test's expected set. Use `"ui_config"`, not
   `"config"` — see the naming-deviation note above.

### Robust benchmarking (R7.3a) — the flow is the trap
- The metaheuristics take `robust=` through `sample_posterior`; the **flow ignores
  it** there (robust selection for the flow happens later in `recommend_recipe`).
  So a "robust flow" row would misrepresent what ran — omit it, or label it explicitly
  as unconditional sampling. Do not silently reuse the non-robust flow cloud under a
  "robust" heading.
- The robust metric is **lower-bound MAE**, not median MAE: robust mode optimizes the
  conformal lower bound, so score `|predictor.predict_interval(sample)[0] − target|`.
  Reporting median MAE under "robust" would look like a regression when it is a
  different, deliberate objective.
- NSGA robust: `run_nsga(..., robust=True)` adds the in-support constraint; expect
  in-support ≈ 100% and a *smaller* front / lower hypervolume than non-robust — that
  contrast **is** the result. Report both.
- Keep the benchmark smoke test fast: assert the new `robust` runs and emit the
  expected columns with a tiny budget; do **not** assert timing/accuracy values (they
  are stochastic — the existing test's contract).

### Coverage ratchet (R7.3b) — measure, don't guess
CI has **no TensorFlow**, so `src/amortized.py` (~130 stmts) is nearly uncovered there
and the flow tests skip. The floor must sit below CI's real number, not a full local
run's ~78%. Measure the no-TF figure before setting it — either read it from the last
CI log's coverage total, or locally run
`pytest --cov=src -p no:cacheprovider --ignore=tests/test_amortized.py` (and let the
other TF-guarded assertions `importorskip`) to approximate the no-TF total, then set
the floor a few points under it. The roadmap's "68" is a *target*, not a mandate —
if the measured no-TF coverage is 71%, 68 is right; if it's 66%, set 63 and note it.

### Registry sliders (R7.4b) — the 8 features are load-bearing
- The core sliders map 1:1 to the model's 8 input features **in PARAM_NAMES order**.
  `slider_specs_view()` must return them in that exact order (iterate `PARAM_NAMES`,
  look each up in the registry — do **not** iterate `dict` order). A test pins
  `[p for p,*_ in slider_specs_view()] == PARAM_NAMES`.
- The slider `min/max` is a **display** range (what the UI lets a user dial), which is
  intentionally *not* the data envelope the optimizers search (`data_envelope()` from
  the training data). Keep them separate; don't "unify" them.
- Regression gate: the existing `tests/test_materials.py` bit-identical view checks
  must still pass — adding a `slider` block to each core record must not perturb
  `carbon_factors_view` / `unit_costs_view` / `densities_view` / `exotics_view`.

### git-lfs (R7.4c)
Lowest-risk path is usually "document and defer": one `git count-objects -vH` and a
`du -sh`, then a README sentence with the number and the ceiling. Only migrate to LFS
if the packed size already exceeds the ceiling — an LFS migration rewrites history and
changes clone behaviour, which is disproportionate for ~11 MB. If you do migrate, add
`.gitattributes` for `models/**` and `docs/images/**` and note the fetch requirement.

### Sequencing & scope
- Order: R7.3c (effort log, trivial) → R7.3b (measure + ratchet) → R7.3a (robust
  benchmark) → R7.4a (session restore, the biggest) → R7.4b (registry sliders) →
  R7.4c (decision). R7.4a and R7.4b are independent; do them in separate commits so an
  AppTest failure localises.
- Each item lands with its regression test in the same commit; update `PAPER.md`
  §8.1/§8.2 only if the robust benchmark changes a quoted number, and `docs/specs/
  README.md` to mark R7.3/R7.4 shipped with an effort-log row.
- Do **not** weaken an acceptance gate to make it pass. If the coverage ratchet can't
  reach the target honestly, set the honest number and say why.
