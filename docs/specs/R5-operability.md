# R5 — Operability: Session Round-Trip Guarantee + Headless CLI Mode

**Status:** Draft · **Effort:** ~0.5–1 d · **Depends on:** R4.2 (ui/ package — the CLI
must import zero Streamlit), R3.3 (mix ticket — the CLI's primary output format)

## Motivation

1. **Session round-trip is guarded by validation, not by a value-level test.** The R3.4
   regression (v1 exports silently dropped `carbon_factors`) shipped precisely because
   nothing asserted that an exported session, re-imported, reproduces the same state.
   `validate_session_state` checks shape, and the AppTest smoke runs the sidebar code,
   but no test asserts the *values* survive. Every future field added to the export is
   exposed to the same failure class until this exists.
2. **Every run currently requires a browser.** The core logic is deliberately headless
   (`src/`), but there is no way to say "evaluate this configuration" from a shell, a
   CI job, a batch study, or another tool. A CLI closes the gap and doubles as the
   integration seam for the procurement/EPD workflows sketched in R6 and
   `docs/BUSINESS_REPORT.md`.

## 1. Session round-trip test (R5.1)

### Design
- AppTest-based (the export/import path is Streamlit-coupled by design):
  1. Run the app; mutate state that lives in the export — at minimum one mix slider
     key (e.g. `cement_A`), one cost, one **carbon factor** (the field that regressed),
     and one exotic dosage.
  2. Capture `get_state_json()` output.
  3. Fresh AppTest run (new session); feed the JSON through the import path
     (`json.load` + `validate_session_state` + the same assignments the sidebar makes —
     factor the sidebar's import body into a `ui.state.apply_session(data)` helper so
     the test and the sidebar share one code path rather than the test re-implementing it).
  4. Assert every exported field equals the mutated value: slider keys, costs,
     carbon_factors, exotics.
- Plus a pure-logic completeness gate that needs no Streamlit: the set of keys in
  `get_state_json()` must equal the set of keys `apply_session` consumes. A field added
  to one side but not the other fails the test — that is the R3.4 class caught at
  edit time.

### Acceptance gates
- Value-level round-trip test green, including a mutated carbon factor.
- Export/import key-set symmetry test green.
- Sidebar import behavior unchanged (existing AppTest smoke stays green).

## 2. Headless CLI mode (R5.2)

### Design
`python -m src.cli <command>` — imports from `src/` only (no `streamlit`, no `ui/`).

| Command | Input | Output |
|---|---|---|
| `predict` | `--mix mix.json` (named params or 8-vector) + `--config project.json` | metrics as JSON to stdout, or `--ticket out.csv` (the R3.3 mix ticket) |
| `design` | `--target 45 [--backend ga\|aco\|flow\|auto] [--age 28] [--robust/--no-robust]` + config | recommended recipe as JSON / ticket |
| `pareto` | `--algorithm nsga2\|nsga3 --pop 60 --gen 40` + config | front as CSV/JSON |

- `project.json` **is the session-export schema** (versioned, already validated by
  `validate_session_state`): costs, carbon_factors, plus a `config` block for
  transport_km / cement_type / advanced / robust / design_age. One schema serves the
  UI session, the CLI, and (later) programmatic callers — no second config dialect.
- All computation goes through the existing single paths (`compute_metrics`,
  `recommend_recipe`, `run_nsga`, `mix_ticket`) — the CLI is wiring, not logic.
- Exit codes: 0 success, 1 validation error (message on stderr), 2 environment
  (missing model artifacts / optional deps).

### Acceptance gates
- `python -m src.cli predict --mix ... --config ...` produces JSON whose strength/
  carbon/cost equal `compute_metrics` called directly (test asserts equality).
- `design --target 45` recipe meets the same target-tracking tolerance as the
  existing `recommend_recipe` tests.
- `python -c "import src.cli"` does not import streamlit (test asserts
  `"streamlit" not in sys.modules`).
- Ticket output parses under the existing mix-ticket tests' assertions.

## Out of scope
- A daemon/REST API (the CLI is the seam; a server can wrap it later).
- Batch/sweep orchestration (compose with shell loops or the R6 registry work first).

## Implementation notes (sonnet-ready)
- **R5.1:** the only trap is the keyed-widget rule from R4 — in the fresh-session step,
  apply the import *before* widgets instantiate (AppTest: set `at.session_state` /
  call `apply_session` prior to `.run()`, mirroring the sidebar's position in the
  script). Do not `at.run()` first and then write slider keys directly.
- **R5.2:** reuse `validate_session_state` and add missing-field defaults rather than a
  new schema. Keep argparse in `src/cli.py` with a `main(argv=None)` entry so tests
  call it in-process (no subprocess needed except one smoke).
