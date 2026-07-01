# Fix Plan: Chemical & Generative Models

**Source of findings:** `docs/MODEL_CRITIQUE.md`
**Sequencing principle:** stop the harm first, build the shared foundation, then make each layer
real bottom-up (forward → chemistry → **simple honest generator** → UI), testing at each gate.
Each phase is independently shippable.

**Critical path:** P0 → P1 → **P3 (simple generative model)**. P2 and P4 can be parallelized once
P1 lands. Phases 0–3 recover *correctness and honesty*; 4–6 are *quality and durability*; P7 is
optional depth.

## Status (2026-07-01)

| Phase | State | Commit |
|---|---|---|
| P0 Safety (import guard, interim banner) | ✅ done | `d335053` |
| P1 Foundation (delete physics.py, consolidate constants) | ✅ done | `9111e17` |
| P2 Chemistry correctness (unify tiers, JSON clinker factors, report fix) | ✅ done | `60c598e` |
| P3 Simple GA generator (`PopulationInverseDesigner`) | ✅ done | `1ce9c59` |
| P4 Retire heuristic planner | ✅ done | `fc4a983` |
| P5 UI honesty relabel | ✅ done | `fc4a983` |
| P6 Regression tests (21 passing) | ✅ done | `fc4a983`, `60c598e` |
| P7 Real amortized BayesFlow flow + SBC | ✅ done | `54dd91e` |
| P7 ACO generator variant | ✅ done | this commit |
| P7 remainder (smoother rejection sampler) | ⏳ optional | — |

---

## Phase 0 — Stop the bleeding (safety, ~½ day)
*Independent. Do first — the app currently can't import without TensorFlow, and Tab 2 shows noise as inference.*

1. **Guard the TF/BayesFlow imports** in `src/bayesian.py` (`try/except`, mirror the existing `bf`
   guard) so `import src.bayesian` and the Streamlit app degrade instead of crashing.
2. **Add an honest interim banner** to app Tab 2 ("Amortized Performance Estimates"): mark it
   experimental and state the sampler is not yet target-conditioned. Temporary; removed in Phase 5.

**Gate:** `python -c "import app"` works with no TF installed.

---

## Phase 1 — Shared foundation (~½ day)
*Unblocks Phases 2–4. No behavior change to the parts that work.*

3. **Consolidate carbon/cost constants** into one module (keep `chemistry_simple`; remove the
   duplicated `CARBON_FACTORS`/`UNIT_COSTS`).
4. **Resolve `physics.py`** — dead + broken (`calculate_embodied_carbon` doesn't exist; it's dead
   code after a `return`). **Recommend deleting it** and repointing the Appendix-C manifest; its only
   working half duplicates `chemistry_simple`.

**Gate:** a single source of carbon/cost constants; no module imports the missing symbol.

---

## Phase 2 — Chemistry correctness (~1 day)
*Depends on Phase 1.*

5. **Unify the two carbon tiers onto the same system boundary** — advanced = clinker chemistry *for
   cement* **+** the same aggregate/water/admixture terms as simple. The tier toggle then changes
   *fidelity, not scope*.
6. **Load `clinker_factor` from `data/oxide_compositions.json`** instead of hardcoding `0.95`
   (unlocks LC3=0.50, slag/ash=0.0 — the biggest carbon lever).
7. **Correct the technical report** §6.3: measured direction is simple ~12–16% *higher* than
   advanced, not lower. Fix after step 5 makes the numbers comparable.

**Gate:** advanced-tier carbon ≥ its own clinker-only term and comparable to simple on a pure-OPC
mix; report matches measured direction.

---

## Phase 3 — Simple honest generative model (GA-based inverse designer) (~1 day) ← the key first step
*Depends on Phase 1/2 (uses the consolidated carbon fn). This replaces the noise sampler as the
interim generative capability and defers the real amortizer to P7.*

**Why this over a rejection sampler or a trained flow:** it reuses the existing
`GeneticOptimizer(objective_fn, bounds).run(n)`, needs no TensorFlow, is transparent to a reader,
and is genuinely conditioned on the target — the three things the current `BayesFlowExplorer` is not.

8. **Add `src/generative_ga.py` — `PopulationInverseDesigner`:**
   - Objective: minimize `|predict(m) − target_strength|`, with an optional additive carbon penalty
     `λ·max(0, carbon(m) − carbon_target)` using the consolidated carbon fn.
   - Search **bounds clamped to the UCI data envelope** (per-column min/max from `load_data`) so
     candidates stay in-distribution — directly fixes the OOD water (<121.8 kg/m³) the old planner emitted.
   - Wrap `GeneticOptimizer` (pop 50–100, ~40 generations).
   - **Generative output = the top-K diverse individuals of the final population**, not just
     `get_best()`. That spread of near-target mixes is the honest stand-in for a posterior.
9. **Expose it behind the existing interface:** give `BayesFlowExplorer.sample_posterior(target, …)`
   a real, target-conditioned implementation that delegates to `PopulationInverseDesigner` (keep the
   signature so `app.py` Tab 2 and `suggest_tests` need no change). Fix `evaluate_uncertainty`
   (currently `np.random.uniform`) — back it with population spread or `models.predict_variance`.

**Gate (target-tracking test):** samples for target=20 vs 80 MPa now differ, and round-trip through
the predictor near their requested targets (the old sampler returned an identical cloud for both).

---

## Phase 4 — Retire the broken inverse planner (~½ day)
*Depends on Phase 3 — the GA designer already closes the loop.*

10. **Replace `inverse_plan_mix`'s heuristic body** with a call to `PopulationInverseDesigner`
    (single best individual for the notebook's one-shot use). Deletes the target-miss behavior
    (25 MPa → 51 MPa) and the water-vs-cement-only bug in one move.

**Gate (envelope test):** planner output stays within UCI ranges and round-trips within tolerance of
the requested strength.

---

## Phase 5 — UI honesty & wiring (~½ day)
*Depends on Phase 3.*

11. Remove the Phase-0 interim banner; the Tab 2 surface now reflects the real target-conditioned
    population, matching its caption.
12. Label heuristic metrics (curing time, transport carbon) as heuristics next to model-backed strength.

---

## Phase 6 — Regression tests & docs (~½ day, parallel with P3–P5)
*Add to `tests/` as each fix lands, not at the end.*

13. Tests: sampler target-tracking, planner envelope-clamp, carbon-tier superset relation,
    `physics` symbol removal, and a smoke `import app`.
14. Update `MODEL_CRITIQUE.md` statuses and report Appendix C.

---

## Phase 7 — Optional depth (later)
*Not required for a correct, honest system.*

15. ✅ **Done.** Added **ACO** (Ant Colony Optimization for continuous domains, ACO_R) as a second
    population generator behind the same interface: `src/aco.py` (`AntColonyOptimizer`, an
    `Optimizer` subclass) + `AntColonyInverseDesigner` sharing the GA designer's plumbing via a
    common base. Selectable as `sample_posterior(..., method="aco")`. Both metaheuristics hit their
    targets in-envelope.
16. Add **importance/rejection sampling** for a smoother posterior surface if the discrete GA
    population looks too sparse in the UI.
17. ✅ **Done.** Trained a **real amortized BayesFlow posterior** (`src/amortized.py`) behind the
    same `sample_posterior` interface, with the **SBC diagnostic** the report admits (§5.2) had
    never been run — all 8 parameters calibrate to a mean normalized rank of ≈0.50. It is the
    default backend when trained weights exist; the GA designer remains the fallback. Fully
    documented in `docs/AMORTIZED_INFERENCE.md`.

---

### Dependency summary
```
P0 (safety) ─┐
P1 (foundation) → P2 (chemistry) ─┐
                                  ├→ P3 (simple GA generator) → P4 (retire planner) → P5 (UI)
                                  │        P6 (tests) runs alongside P3–P5
                                  └────────────────────────────────────→ P7 (ACO / rejection / real flow)
```
