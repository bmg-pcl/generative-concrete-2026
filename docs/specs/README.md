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

---

## Forward roadmap

### Horizon 1 — R7: Trustworthy to the edge (~3–4 d)

The theme: the robust mode is the flagship, so its foundations must be beyond
reproach, and the amortized flow must work under the *default* configuration
(today a pinned age routes every query to the GA, so the flow — the most
interesting model in the repo — serves almost nothing).

**R7.1 Joint distributional strength model** *(the §8.3 fix — do first)*
- **What:** replace the separately-trained mean + quantile XGBoost pair with one
  monotone multi-quantile (or distributional-boosting) model; the median becomes the
  point prediction; CQR stays on top unchanged.
- **Why now:** the observed mean-below-lower-bound crossing (`../PAPER.md` §8.3) is
  a *coherence* defect in the flagship robust mode: the display can be incoherent
  and the robust objective can be gamed where the two models disagree. Every H2
  property model will inherit this architecture — fix it before multiplying it.
- **Value:** robust recommendations whose point estimate, interval, and optimized
  bound come from one distribution; §8.3 moves from "known pathology" to "resolved".
- **Gates:** zero crossings on the test fold and a 10k-sample envelope grid;
  coverage within [0.88, 0.92]; point-prediction RMSE within 5% of current;
  benchmark table regenerated.

**R7.2 Flow v2 — (strength, age) conditioning** *(unlocks the default path)*
- **What:** retrain the BayesFlow posterior conditioned on (target strength, design
  age); update routing so pinned-age queries use the flow; SBC per dimension across
  a (strength × age) grid. (This is the item R2.1 explicitly escalated.)
- **Why now:** fixed age is the *default* configuration; under it the amortized
  backend — 10× faster, genuinely posterior-diverse — is dead code. Highest
  capability-per-day item on the board, and a prerequisite for H3's joint posterior.
- **Value:** default-config users get instant, calibrated design clouds; the
  benchmark's flow rows become representative instead of hypothetical.
- **Gates:** SBC normalized ranks uniform (KS test not rejected at 0.05) across the
  grid; age honored exactly in samples; benchmark row with age pinned; staleness
  fingerprint extended to the conditioning schema.

**R7.3 Evidence hardening for the flagship**
- **What:** benchmark the *robust* mode (lower-bound MAE, in-support %, front
  hypervolume for NSGA) alongside the current non-robust table; ratchet the CI
  coverage floor 60 → 68; record actual-vs-estimate effort per spec from now on.
- **Why now:** §8.1's table currently evidences the mode nobody should default to.
- **Value:** the recommended configuration is the measured one; estimates start
  teaching us something.
- **Gates:** BENCHMARKS.md gains the robust section; CI green at 68%.

**R7.4 Debt sweep (small, bounded)**
- Full session restore for the remaining Config widgets (transport, cement source,
  chemistry toggle — the R3.4 deferral) via the established keyed-widget pattern;
  core sliders generated from the registry (the R6.4 deferral), making
  `materials.json` the single authority including UI; decide git-lfs before
  `docs/images/` + models outgrow plain git.
- **Gates:** round-trip test extended to every Config field; a registry dosage-bound
  edit changes a slider with no code change.

### Horizon 2 — R8: From strength tool to specification tool (~1–2 wk)

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

**R8.3 Carbon intervals** — propagate the registry's per-factor `uncertainty`
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

The per-spec **Implementation notes (sonnet-ready)** convention continues. For H1:
- **Ready with care:** R7.3, R7.4 (the keyed-widget trap notes in R4/R5 apply
  verbatim; the registry→slider generation must keep the 8 model features fixed).
- **Senior / escalate:** R7.1 (model replacement — the coverage and no-crossing
  gates are the net, but architecture choice needs judgment) and R7.2 (flow retrain
  was already escalated once from R2.1; SBC-across-a-grid acceptance needs care).
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
