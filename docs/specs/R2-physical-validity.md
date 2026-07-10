# R2 — Physical Validity

**Status:** Draft · **Effort:** ~1–2 days · **Depends on:** R1.2/R1.3 only for the shared
NSGA constraint plumbing (age fix can start immediately)

## Motivation

The generators can currently propose designs that are not valid concrete:

1. **Age is a free gene and the optimizers exploit it.** Observed designer outputs hit
   strength targets with `age=343` / `age=365` — "achieving 65 MPa" partly by prescribing
   a one-year cure. Strength targets are conventionally specified at a fixed age (28 d);
   age is a *condition*, not a design variable.
2. **No volume balance.** Real per-m³ mixes satisfy Σ(massᵢ/ρᵢ) + air ≈ 1 m³. Every UCI
   row does; the box envelope does not — corner and many interior combinations are
   physically unbatchable. Novelty only *flags* the extremes.
3. **Workability is unmodeled.** A front-optimal mix that cannot be placed is a failed
   design (acknowledged in `TECHNICAL_REPORT.md` §1.3.2, enforced nowhere).

## 1. Age as a fixed condition

### Design
- `generative_ga._InverseDesignerBase` and `nsga.run_nsga` accept `age: float = 28.0`.
  Implementation: clamp the age dimension's bounds to `[age, age]` (degenerate bound —
  note `amortized.py` already guards zero-width std, and pymoo handles `xl == xu`).
- **UI:** "Design age (days)" number input (default 28) on the Inverse Design and Pareto
  tabs; threaded to designers/NSGA. Compare Mixes keeps its free age slider (forward
  prediction at any age is legitimate).
- **Amortized flow:** the current flow conditions on strength only, so its samples roam
  age. Final design: retrain with a 2-D conditioning vector `(strength_z, age_z)` and a
  7-parameter output (drop age from θ; append the conditioned age when assembling the
  8-vector). Rerun SBC; bump the artifact (staleness fingerprint changes with the file
  format — extend `_norm.npz` with a `format_version` key). **Interim:** when the user
  fixes age, `sample_posterior(method="auto")` routes to the GA designer with the age
  clamp and the UI says why; `method="flow"` warns that the flow ignores the age setting.

### Acceptance gates
- Test: GA/ACO `design(45, age=28)` returns mixes with `age == 28` exactly; NSGA front
  likewise.
- Test: recommended recipe at fixed age hits the target within tolerance (proves the
  optimizer no longer needs the age cheat).
- AppTest smoke stays green.

## 2. Volume-balance constraint

### Design
- New `src/physical.py` (name is free — the old broken `physics.py` was deleted):
  - `DENSITIES` (kg/m³): cement 3150, slag 2900, ash 2300, water 1000,
    superplasticizer 1100, coarse_agg 2700, fine_agg 2650 — sourced constants with
    citations in comments; `AIR_FRACTION = 0.02`.
  - `mix_volume(mix) -> float` (m³ per nominal m³ batch): Σ massᵢ/ρᵢ + air.
  - `volume_error(mix) -> float`: `|mix_volume − 1.0|`.
  - `repair_volume(mix) -> mix`: scale the aggregate masses (coarse+fine, preserving
    their ratio) to close the balance; fall back to scaling all masses if aggregates
    alone cannot close it within envelope bounds.
- **Calibrate the tolerance against the data first:** compute `volume_error` over all
  1,030 UCI rows; pick `tol` covering ≥ 90% of rows (expected ≈ 0.03–0.05 m³ given
  density assumptions). Record the measured distribution in the spec's implementation PR.
- Enforcement: NSGA second inequality constraint `volume_error − tol ≤ 0`; GA/ACO
  objective penalty `+ w_vol · max(0, volume_error − tol)`; samplers (`.sample()` cloud
  and the flow's output) pass through `repair_volume` before clipping. Compare Mixes
  shows a warning when the hand-set A/B mix violates the balance.

### Acceptance gates
- Test: ≥ 90% of UCI rows pass `volume_error ≤ tol` (validates the density table).
- Test: all-max-corner mix fails; `repair_volume` of it passes and stays in-envelope.
- Test: NSGA front and designer clouds satisfy the balance.

## 3. Workability guard (heuristic first)

### Design
- `workability_flag(mix) -> Optional[str]` in the same module: returns a warning string
  when `w/b < 0.35` **and** `superplasticizer < 6 kg/m³` ("likely unplaceable without
  more superplasticizer"), or `w/b > 0.65` ("segregation risk"). Displayed on Compare
  cards and the recommended recipe; **not** a hard constraint in v1 (heuristic
  confidence is low).
- Stretch (separate PR): train a slump model on the UCI slump dataset already listed in
  `data_fetcher.AVAILABLE_DATASETS` (103 rows — label it low-data), and replace the
  heuristic flag with a predicted slump range.

### Acceptance gates
- Unit tests for the flag's boundary cases; flags render (AppTest can assert warning
  presence for a known-bad mix).

## Out of scope
- Rheology modeling, fiber effects on workability, pumping simulations.

## Risks
- Density constants vary by source material; tolerance calibration against the dataset
  (step 2) is what keeps the constraint from rejecting real mixes. If UCI pass-rate is
  below ~80% with standard densities, widen `tol` and document rather than tune densities
  to fit.
- Fixing age narrows the achievable strength range at 28 d (no more 365-day cheats);
  targets near the top of the range will honestly report larger misses — this is the
  point, but the UI should say so.

## Implementation notes (sonnet-ready)

- **ESCALATE the flow retrain — do not attempt it to "complete" R2.1.** The "final
  design" (BayesFlow θ 8→7, 2-D `(strength, age)` conditioning, re-standardisation, SBC
  rerun, artifact `format_version`) is a research-grade change with many silent-failure
  points; it is **out of scope for a sonnet-class implementer**. Ship the **interim
  only**: add the `age` clamp to GA/ACO/NSGA, and when age is user-fixed route
  `method="auto"`/`"flow"` to the GA designer with a UI note ("the trained flow ignores
  the age setting — using the GA designer"). Leave a `TODO(flow-age-conditioning)` and
  file the retrain as a separate senior task.
- **Age clamp mechanics:** pass `age` into the designer/`run_nsga`; override *only* the
  age dimension's `(xl, xu)` to `(age, age)`. `uniform(a, a) == a`, so GA init and
  mutation collapse correctly; pymoo accepts `xl == xu`. Don't special-case age elsewhere.
- **Volume tolerance is data-driven, not guessed:** compute `volume_error` over all 1,030
  UCI rows FIRST, print the distribution, set `tol` to the 90th percentile. If the
  pass-rate with the density table is < 80%, **widen `tol` and document it — do NOT tweak
  densities to fit.** Record the measured distribution in the PR description.
- Do R2.2 before R2.3; the workability flag is a pure heuristic and independent.
