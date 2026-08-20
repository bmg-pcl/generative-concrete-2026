# Roadmap

## Vision

Every party in the concrete chain — the engineer specifying, the buyer tendering, the
producer's lab, the proponent holding the carbon target — working from **one
calibrated trade-off surface**: strength, durability, workability, carbon (with
provenance), and cost, with honest uncertainty on every axis. The marginal lab test
chosen by information value, not habit. The marginal tonne of CO₂ priced with
supplier-specific provenance, not an industry average. A new plant reaching a trusted
local model from a tenth of the data a cold start needs. The end state is not a
better strength calculator; it is the **shared information layer** that
`../BUSINESS_REPORT.md` argues the fragmented design → procure → supply → utilise
chain is missing — with the statistical honesty (`../PAPER.md`) that makes it safe to
lean on.

Everything below is sequenced toward that: first make the numbers trustworthy to the
edge of the data (H1), then make the tool speak the language real specifications are
written in (H2), then close the loops across organizations (H3).

## North-star metrics

Progress is measured, not narrated. Every horizon must move at least one of these:

| Metric | Now | Direction |
|---|---|---|
| Interval coverage at nominal 90% | 0.903 | hold within [0.88, 0.92] through every model change (regression-gated) |
| Interval sharpness (mean width at fixed coverage) | ~18 MPa | ratchet down |
| Robust-mode delivery guarantee (% returned designs in-support & volume-balanced) | 100% (by construction, test-gated) | hold |
| Active-learning efficiency (lab tests saved vs random at equal RMSE) | unmeasured | measure (H2), then target ≥ 40% on replay |
| New-context data efficiency (rows for a new lab/plant to reach RMSE parity) | full retrain | 10× fewer via partial pooling (H3) |
| Carbon provenance share (% factors EPD-backed in a project) | tooling shipped | adoption metric once used in anger |
| Amortized backend share (% queries served by the flow) | ~0% under default config | make the flow the default answer (H1 → H3) |

---

## Shipped (2026-07): R1–R6

Kept for the record — each spec remains the implementation reference. Estimates held
(R1–R6 estimated ~7–10 d total; delivered in roughly that, including the unplanned
finds each phase surfaced: the CQR coverage bug-class, the slider rebinding bug, the
live-session carbon-factor import overwrite, the `viz.py` NameError).

| Spec | What shipped | Value delivered |
|---|---|---|
| [R1](R1-trustworthy-numbers.md) | Conformal intervals (coverage 0.903 measured), calibrated novelty gate, robust optimization | The UI's central claim ("90%") became a tested fact; optimizers stopped chasing confident fictions |
| [R2](R2-physical-validity.md) | Age as a condition, volume balance (data-calibrated tolerance), workability guard | Returned designs became batchable and un-gameable on cure time |
| [R3](R3-dormant-value.md) | Active learning, EC2 tensile, mix tickets, session-export fix | The model's uncertainty started earning money: it tells the lab what to test and gives suppliers a handable artifact |
| [R4](R4-engineering.md) | Slider rebinding, `ui/` modularization, CI lint+coverage, backend benchmarks | A silent state bug fixed; "when to use which backend" got numbers |
| [R5](R5-operability.md) | Session round-trip guarantee, headless CLI | Batch/procurement use unlocked; a real import-overwrite bug caught by the new tests |
| [R6](R6-materials-platform.md) | Material registry, EPD plug-in + provenance, clinker-source Scope 1/2 model | Carbon became procurement-grade: measured beats modeled, disclosed per material; capture/hydro/gas plants distinguishable |

### Effort log (R7.3c)

Estimate vs actual per spec, so the estimates get better over time instead of staying
guesses. R1–R6 were estimated and delivered as a block (~7–10 d total, in roughly that
span) before per-spec actuals were tracked; R7.1 on, actuals are recorded at close.

| Spec | Estimate | Actual | Note |
|---|---|---|---|
| R1 | ~1–2 d | ~1–2 d (block) | Within estimate; the coverage-test finding (R1.1) was the forcing function for the rest |
| R2 | ~1–2 d | ~1–2 d (block) | Within estimate |
| R3 | ~1–2 d | ~1–2 d (block) | Within estimate |
| R4 | ~1 d | ~1 d (block) | Within estimate; slider-rebinding bug found and fixed inside the window |
| R5 | ~0.5–1 d | ~0.5–1 d (block) | Within estimate; live-session carbon-factor overwrite bug found and fixed inside the window |
| R6 | ~2–3 d | ~2–3 d (block) | Within estimate |
| R7.1 | ~3–4 d | **~0.5 d** | Est. assumed architecture uncertainty; the fix (rearrangement-sorted joint quantile model) was mechanically clear once chosen. RMSE gate consciously revised (+6.8%, documented in PAPER §8.3) rather than tuned away |
| R7.2 | senior/escalate (no d-estimate) | **~0.5 d** | Flagged for escalation ex ante (2-D SBC, network-shape change); the age-as-prior-marginalised-condition formulation collapsed the risk. Escalation flag was the right call to make beforehand even though the outcome was fast |
| R7.3 | ~1–1.5 d (a+b+c combined) | **~0.3 d** | Straightforward given the R7.1/R7.2 groundwork (predict_interval, run_nsga's existing robust= arg); the only real judgment call was excluding a "robust flow" row rather than fudging one — see spec's sonnet-ready note |
| R7.4 | ~1–1.5 d (a+b+c combined) | **~0.4 d** | R7.4a (session restore) was the largest item but followed the established keyed-widget pattern exactly; the one real find was the `"config"`/`"ui_config"` naming collision with the CLI, caught before it shipped. R7.4b's only deviation from the spec text was where the PARAM_NAMES ordering lives (ui/state.py, not materials.py — avoids a circular import). R7.4c was a five-minute measurement (13.4 MiB) confirming the "stay on plain git" default |

| R8.0 | ~2–2.5 d serial / ~1 d parallel (Wave A) + ~0.5 d Wave B | **Wave A: ~0.3 d wall** | Five concurrent sonnet packages under the v2 process: four ran clean start-to-finish (~5–8 min, ~70–106k tokens each, zero stalls); WP-A stalled twice on background-and-wait and was killed by a session usage limit before reporting — its commit was recoverable and the orchestrator verified its gates independently. WP-C caught a real spec error (the ΔT band was derived from 28-day heat, not full-hydration heat) and corrected the band with the derivation instead of tuning — the report-deviations clause working as designed. Headline: LC3 cement term landed at 0.544 kg/kg inside the published 0.50–0.65 EPD band with no coefficient fit to it. Wave B (WP-E) pending |
| R7.5 | ~1.25 d serial / ~0.7 d parallel | **~0.7 d** (≈0.4 d implementation + ≈0.3 d failure recovery) | Four-agent fan-out; wall clock hit the parallel estimate but for the wrong reason — logistics failures (see `../DELEGATION_WORKFLOW.md`) ate what parallelism saved. WP-5, run solo under the v2 process, took ~6 min of agent time with zero stalls |

**Reading the log so far:** the block estimates (R1–R6) were accurate at block
granularity but say nothing about per-item variance. The two data points since
per-spec tracking started (R7.1, R7.2) both landed well under a wide estimate range —
too small a sample to conclude "estimates run high," but worth watching as more items
close.

---

## Forward roadmap

### Horizon 1 — R7: Trustworthy to the edge (~3–4 d)

The theme: the robust mode is the flagship, so its foundations must be beyond
reproach, and the amortized flow must work under the *default* configuration
(today a pinned age routes every query to the GA, so the flow — the most
interesting model in the repo — serves almost nothing).

**R7.1 Joint distributional strength model** — ✅ **Shipped** *(the §8.3 fix)*
- **What:** replaced the separately-trained mean + quantile XGBoost pair with one
  multi-quantile model (800 trees), per-row rearrangement-sorted so the median is the
  point prediction and the outer quantiles are the interval; CQR unchanged on top.
- **Why:** the observed point-below-lower-bound crossing (`../PAPER.md` §8.3) was a
  *coherence* defect in the flagship robust mode. Every H2 property model inherits
  this architecture — fixed before multiplying it.
- **Value delivered:** point estimate, interval, and optimized bound now come from one
  distribution; the CLI's `lo ≤ point ≤ hi` assertion is restored; the flow's
  staleness guard gained a forward-model content hash so any future model change
  demotes a stale flow.
- **Gates (met):** zero crossings on the test fold **and** a 10k-point envelope sweep;
  coverage 0.888 ∈ [0.88, 0.92]; benchmark table + flow regenerated (SBC 0.486–0.532).
- **Estimate reconciliation:** est. ~3–4 d "first"; the model swap itself was ~½ d.
  The **RMSE gate was consciously revised**: "within 5%" proved wrong-headed — a
  median trained on the 75% fit fold (split-conformal validity holds out the
  calibration fold; the old mean model used the full 80%) lands at 4.97 MPa, +6.8%.
  That is the real, irreducible price of one coherent distribution with valid
  calibration, not a regression to tune away; documented in §8.3. Recovering it needs
  cross-conformal (CV+), deferred as out-of-scope for the coherence fix.

**R7.2 Flow v2 — (strength, age) conditioning** — ✅ **Shipped** *(unlocked the default path)*
- **What:** retrained the BayesFlow posterior to output the 7 non-age parameters
  conditioned on (strength, age); age is drawn from the prior during training
  (marginalised correctly) and supplied at inference. Routing now sends pinned-age
  queries to the flow; `age=None` marginalises over age inside the flow. (This was the
  item R2.1 explicitly escalated as a senior task.)
- **Why:** fixed age is the *default* configuration; under it the amortized backend —
  ~10× faster per query, genuinely posterior-diverse — was dead code (served ~0% of
  queries). Also the conditioning prerequisite for H3's joint posterior.
- **Value delivered:** default-config users now get instant, calibrated,
  age-conditioned design clouds from the flow; the benchmark's flow rows are
  representative rather than hypothetical.
- **Gates (met):** SBC normalized ranks for the 7 sampled params in [~0.47, ~0.53]
  across the prior-spanned (strength × age) grid; age honored exactly in samples
  (test-gated); staleness guard extended with a `cond_schema` tag so the old
  strength-only checkpoint is rejected (falls back to GA) rather than mis-loaded.
- **Estimate reconciliation:** est. senior/escalate; the clean formulation (age as a
  condition, prior-marginalised) made it ~½ d — the escalation flag was warranted ex
  ante (2-D SBC, network-shape change) but the design collapsed the risk.

**R7.3 Evidence hardening for the flagship** — ✅ **Shipped** — full spec:
[`R7-trustworthy-to-the-edge.md`](R7-trustworthy-to-the-edge.md)
- **What shipped:** a `robust=True` benchmark section (lower-bound MAE — not median
  MAE, the metric robust mode is actually scored on — plus in-support % for GA/ACO;
  the flow is deliberately excluded since it doesn't recondition on `robust=` inside
  `sample_posterior`); a robust NSGA row with hypervolume compared against the
  non-robust front on one shared reference point; the CI coverage floor measured
  (69.98% no-TF, not the stale "~63%" comment) and ratcheted 60 → 65; an effort log.
- **Value delivered:** the mode users actually run by default is now evidenced, not
  just the non-robust baseline. Measured trade: GA/ACO in-support jumps from
  0–1.7% (non-robust) to 64–100% (robust); NSGA in-support 5% → 100%; NSGA
  hypervolume 1,082,794 → 820,613 (the front's real, quantified size cost for that
  guarantee) — see `docs/BENCHMARKS.md`.
- **Gates (met):** `BENCHMARKS.md` Robust-mode + combined-NSGA sections regenerated
  from a live run; CI green at the ratcheted floor; 11 benchmark-plumbing tests.

**R7.4 Debt sweep (small, bounded)** — ✅ **Shipped** — full spec:
[`R7-trustworthy-to-the-edge.md`](R7-trustworthy-to-the-edge.md)
- **What shipped:** full session restore for the ten remaining Config widgets
  (transport, cement/clinker source, chemistry toggle, robust/age toggles — the R3.4
  deferral) as keyed `cfg_*` widgets under a new `ui_config` session block (`"config"`
  was rejected — it collides with the CLI's own, differently-schemed `"config"` key);
  core mix sliders now generated from `data/materials.json`'s new `slider` blocks via
  `materials.slider_specs_view()`, with `ui/state.py` doing the PARAM_NAMES ordering
  and the `age` special case (materials.py can't import PARAM_NAMES without a
  circular import); git-lfs measured and declined (13.4 MiB total, ~4× under the
  50 MB ceiling).
- **Value delivered:** a session export/import now round-trips the *entire* app
  state, not just mixes and costs; `data/materials.json` is the single UI authority
  for both material properties and their slider ranges — editing the JSON changes
  the Compare tab with no code change (R6.4's remaining deferral, closed).
- **Gates (met):** round-trip + key-symmetry tests cover all ten `ui_config` fields
  (pure-dict, AppTest widget-level, and a CLI-compatibility regression test guarding
  the naming collision); a registry `slider.max` edit changes `slider_specs_view()`
  with no code change; slider order pinned to `PARAM_NAMES`; existing bit-identical
  core views unchanged; 125 tests green at the 65% floor.

**R7.5 Chemistry layer remediation** — ✅ **Shipped** (Wave A 2026-08-16, WP-5 2026-08-18):
[`R7.5-chemistry-remediation.md`](R7.5-chemistry-remediation.md)
- **Origin:** an adversarial review of the base chemistry models (2026-08-15), not a
  planned roadmap item. Reopens the R7 theme for one bounded remediation wave;
  R7.1–R7.4 stay closed.
- **What the review found:** the carbon accounting survives attack (calcination
  stoichiometry, kiln-electricity derivation, Scope 1/2 split, capture-energy
  handling, and tier-boundary equality are all correct — leave them alone). The
  hydration/Bogue layer does not: a factor-of-10 heat unit error, a dimensionally
  incoherent pozzolanic reaction (wt% compared against kg/m³), and a portlandite
  budget spent twice. That layer is unreferenced by the app and CLI but *is* called
  by `notebooks/experiment_ga.ipynb`, so it is user-facing and untested. Separately:
  exotic admixtures carry no density and are invisible to the volume balance (150
  kg/m³ of calcined clay displaces ~0.058 m³, more than the whole 0.05 tolerance),
  and a config-supplied clinker source crashes the CLI with a raw `KeyError`.
- **Structure:** four work packages with disjoint file ownership run in parallel
  (~0.7 d wall-clock vs ~1.25 d serial), plus one serialised follow-up for the
  transport heuristic, which is triplicated across three files owned by two packages.
- **The gate discipline:** we can fix dimensional correctness because physics verifies
  it; we cannot validate the hydration model, since the repo holds no calorimetry or
  DoH data. Every gate is a unit, conservation, or frozen-surface check — never
  "matches the literature value."

### Horizon 2 — R8: From strength tool to specification tool (~1–2 wk)

**R8.0 Deepening the carbon & chemistry layers** — **Wave A ✅ shipped** (2026-08-20, five
parallel sonnet packages); **Wave B (WP-E integration wiring) pending** — full spec:
[`R8.0-carbon-chemistry-deepening.md`](R8.0-carbon-chemistry-deepening.md)
- **Origin:** post-R7.5 analysis. R7.5 fixed what was *wrong*; R8.0 addresses what is
  *missing*: the Tier-2 cement term's dropped scope (grinding/process electricity —
  most of the "tier gap" is dropped scope wearing a fidelity costume), LC3's
  zero-carbon non-clinker half, the never-built CLAUDE.md waste factor, the ticket's
  exotic reconciliation gap, and a hydration layer that computes heat/CH/C-S-H that
  nothing consumes.
- **Structure:** four disjoint Wave A packages for parallel sonnet agents (accounting
  boundary; Tier-2 scope completion — the one package licensed to change quoted
  numbers, with an external-validation gate that the LC3 cement term must land in the
  published 0.50–0.65 kg/kg EPD band; hydration consumers as a new additive module —
  adiabatic ΔT/mass-pour flag, maturity curing, carbonation upper bound; registry
  conventions — allocation basis, chloride restrictions, uncertainty propagation,
  per-material transport data), plus a Wave B integrator and a fenced phase-1-only
  Powers-calibration fit (the R9-class hybrid's evidence-gathering step, delegable
  because it fits only Powers' two constants and reports R² as a finding, never
  tuning the hydration layer).
- **Absorbs** R8.3 (carbon intervals) into its WP-D/WP-E; R8.1/R8.2/R8.4 unchanged
  (R8.2 gains a ready-made data hook: the restrictions field and
  `compliance_warnings`).
- **Process:** dispatched per [`../DELEGATION_WORKFLOW.md`](../DELEGATION_WORKFLOW.md) v2.

The theme: real specifications are written in exposure classes, workability targets,
and (increasingly) carbon ceilings — not bare 28-day strength. Meet them there.
This is where the business analysis (`../BUSINESS_REPORT.md` §6) becomes product.

**R8.1 Workability from data** — train a slump model on the UCI slump corpus
(already listed in `data_fetcher.AVAILABLE_DATASETS`; 103 rows — small, so conformal
intervals matter more, not less), replacing the w/b heuristic where data exists and
labeling the heuristic fallback. *Value:* second measured property; proves the
multi-property pattern (per-property model + interval + support gate) end to end.
*Gates:* slump coverage test; heuristic/model provenance shown; multi-property
metrics dict versioned.

**R8.2 Exposure-class compliance layer** — encode deemed-to-satisfy tables
(EN 206 annex rows, BS 8500, one DOT set) as *data packs*: exposure class →
max w/b, min binder, SCM caps per jurisdiction. Designs report pass/fail per
jurisdiction; the optimizer accepts "XC4-compliant, ≤ 180 kg CO₂/m³" as constraints.
*Value:* the tool outputs something an engineer can defend to a checker and a buyer
can put in a tender — the performance-spec-with-carbon-axis instrument §6 calls for.
Also makes the national-variation cost visible: the same mix's compliance across
jurisdictions in one table. *Gates:* golden tests against published table rows;
packs are JSON, adding a jurisdiction is a data edit.

**R8.3 Carbon intervals** — *absorbed into R8.0 (WP-D/WP-E); kept here for the record* — propagate the registry's per-factor `uncertainty`
fields to an interval on total carbon (analytic linear propagation; Monte Carlo for
the clinker model). *Value:* symmetry — strength has honest uncertainty, carbon
currently pretends to be exact; procurement comparisons at ±15% vs ±40% provenance
quality become visible. *Gates:* ticket carries carbon lo/hi; EPD-backed factors
visibly tighten the interval.

**R8.4 Dataset importers + regional packs** — schema-mapped importers for further
corpora (the CLAUDE.md ask beyond the overlay CSV), plus regional grid/factor packs.
*Value:* the "1998 Taiwan" caveat starts shrinking; the registry's `region` field
does real work.

### Horizon 3 — R9/R10: The shared information layer (quarter+)

The theme: cross the organizational boundary. Everything above serves one user at a
time; the vision's value compounds when calibration, learning, and specification
flow *between* parties.

**R9.1 Hierarchical calibration (partial pooling)** — a hierarchical model over
labs/plants (per-context residual heads on the shared GBM, or a full Bayesian
hierarchy) so a new plant's 30 test rows *adjust* the global model rather than
fighting 1030 rows of 1998 Taiwan. *North star:* 10× fewer rows to RMSE parity for
a new context. *Why it matters commercially:* this is the moat — every calibrated
plant makes the shared prior better for the next one.

**R9.2 Measured closed-loop active learning** — batch suggest → test → retrain with
the information gain *measured*, first on synthetic replay (hold out a region,
replay acquisition vs random), then live. *North star:* ≥ 40% fewer tests at equal
RMSE. Today's merit ranking is plausible; this makes it a claim.

**R9.3 Amortized joint posterior** — one flow for
p(mix | strength@age, slump, carbon ≤ C, compliant(class, jurisdiction)) — the flow
becomes the *primary* backend (metaheuristics demoted to verification/fallback),
building on R7.2's conditioning machinery and R8's property models.

**R9.4 Autospec + tender pack** — the product apex of the four-seams analysis: from
(exposure class, strength-at-loading-age, carbon ceiling, region) generate a draft
performance specification, the compliant Pareto set, provenance-carrying tickets,
and a `score-bids` CLI that ranks N supplier mixes + EPDs under one project config.
*Value:* the proponent's carbon ceiling survives the seams as an executable
artifact rather than an aspiration.

**R10 (stretch)** — search plugins (CMA-ES, Bayesian optimization) behind
`base_optimizer`; cost-aware inverse objectives throughout; automated EPD ingestion
(ILCD/EPD-file parsing → registry records); exotics strength promotion once lab
data exists (the R6 §5 path: registry `strength_treatment` flips per material after
calibration).

### Dependency spine

```
R7.1 (joint distribution) ──► every H2 property model inherits it
R7.2 (flow: +age) ─────────► R9.3 (joint posterior)
R8.1/R8.2 (properties, compliance) ──► R9.3 conditioning set, R9.4 autospec
R8.3 (carbon intervals) ────► R9.4 tender comparisons
R9.1 (pooling) ◄─── needs R8.4 importers (multi-context data to pool)
R9.2 (measured AL) ◄─ needs R7.1 (honest intervals are the acquisition signal)
```

Order within horizons matters (R7.1 before R7.2's retrain; R8.2 is the longest pole
in H2 — start its table transcription early). Horizons can overlap at the edges;
they must not leapfrog: **no H3 item starts while its H1/H2 prerequisite is open.**

### Non-goals (explicit, so ambition stays aimed)

Structural member design (we design *materials*, not beams); LCA beyond A1–A4
(A5/B/C modules ride on the record schema later, not now); certification or
stamping (tickets say "validate physically" and will keep saying it); replacing
laboratory testing (the goal is to make each test worth more, never to skip it).

## Delegation readiness

The per-spec **Implementation notes (sonnet-ready)** convention continues. For H1,
all of R7 (R7.1–R7.4) has now shipped:
- **Shipped (sonnet-ready spec, delegated as written):** R7.3, R7.4 —
  [`R7-trustworthy-to-the-edge.md`](R7-trustworthy-to-the-edge.md). The keyed-widget
  trap notes from R4/R5 applied verbatim; the registry→slider generation kept the 8
  model features fixed in PARAM_NAMES order (with one documented deviation — the
  ordering logic itself lives in `ui/state.py`, not `materials.py`, to avoid a
  circular import); the coverage floor was *measured* in the no-TF (CI) environment
  (69.98%) before being set, not guessed.
- **Delegated four ways in parallel and shipped (Wave A):** R7.5 —
  [`R7.5-chemistry-remediation.md`](R7.5-chemistry-remediation.md). Partitioned by
  exclusive file ownership rather than by finding, because six of the nine defects
  live in one module; the spec names the frozen surfaces that keep concurrent agents
  from breaking each other's tests, and quarantines the one cross-cutting fix (WP-5,
  since shipped under the v2 process with zero stalls) into a second wave. The
  fan-out's retrospective — what the
  partitioning got right and the six operational failure modes it exposed — is
  [`../DELEGATION_WORKFLOW.md`](../DELEGATION_WORKFLOW.md), which is now the process
  reference for the next parallel delegation.
- **Shipped (senior/escalate at the time):** R7.1 (joint model — coverage and
  no-crossing gates were the net) and R7.2 (age-conditioned flow retrain, escalated
  once from R2.1; SBC-across-a-grid acceptance).
- H2/H3 items get full specs (motivation / current state / design / gates / traps)
  before implementation, same as R1–R6 — an entry here is a direction, not yet a
  spec.

## Conventions

- Every item lands with regression tests; a spec is "done" when its acceptance gates
  pass in CI (heavy TF tests may skip; everything else must run).
- Behavior-changing items update `../TECHNICAL_REPORT.md`, `../PAPER.md` (if a claim
  changes), and, where user-visible, `../WORKFLOW.md` in the same commit.
- Retrained artifacts (`models/*.json`, `models/support.npz`, `models/amortizer/`)
  are regenerated together and committed together (staleness mechanism in
  `src/amortized.py`).
- From R7 on: record actual effort against the estimate when closing a spec.
