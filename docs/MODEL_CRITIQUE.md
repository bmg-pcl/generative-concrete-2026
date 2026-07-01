# Critique: Chemical & Generative Models

**Date:** 2026-06-28
**Scope:** `src/chemistry_simple.py`, `src/chemistry_advanced.py`, `src/physics.py` (chemical);
`src/bayesian.py`, `src/chemistry_advanced.inverse_plan_mix` (generative).
> **Update (2026-07-01):** The critical and high-severity findings below have since been
> addressed — see `docs/FIX_PLAN.md` for the phased work. In brief: `physics.py` deleted;
> `sample_posterior`/`inverse_plan_mix` replaced by a transparent GA inverse designer
> (`src/generative_ga.py`) that is genuinely target-conditioned and stays in-distribution;
> TF import guarded; carbon tiers unified on one system boundary with JSON-driven clinker
> factors; report claim corrected. 14 regression tests pass. The text below is preserved as the
> original point-in-time assessment.

**Method:** Models were initialized and executed end-to-end (venv via `uv`, core stack
`numpy/pandas/scikit-learn/xgboost/openpyxl/xlrd`). The XGBoost forward model was retrained
(RMSE 4.65, R² 0.92 — matches the technical report). Each chemical and generative routine was
run on representative mixes; findings below are reproduced from live output, not read-through.

---

## TL;DR

The **forward stack is sound** (XGBoost predictor, the two chemistry tiers as *estimators*).
The **generative stack is non-functional** and, worse, **presented to users as if it works**:

| Component | Status | Severity |
|---|---|---|
| XGBoost strength predictor | Works; metrics match report | — |
| `chemistry_simple` (carbon/cost/curing) | Works as a linear estimator | Low |
| `chemistry_advanced` (Bogue/hydration/clinker carbon) | Runs; reasonable but uncalibrated | Medium |
| `physics.py` | **`calculate_embodied_carbon` does not exist** — dead code after a `return` | High (latent) |
| `bayesian.BayesFlowExplorer` | **Stub** — `train()` is a no-op, posterior ignores the target | Critical |
| `inverse_plan_mix` | Runs but **misses targets badly** and emits out-of-distribution mixes | High |

---

## 1. Generative models

### 1.1 `BayesFlowExplorer` is a stub dressed as a trained model — CRITICAL

This is the headline feature of the project (README: "BayesFlow amortized inference",
"full posterior probability mesh"). In its current state it does none of that.

**Evidence (run directly):**

- `sample_posterior(target_strength)` **never reads `target_strength`**. Both the trained and
  untrained branches sample `np.random.normal(self.bounds.mean(axis=1), …)`. Asking for 20 MPa
  and 80 MPa returns the *identical* cloud:
  ```
  mean sample @target=20: [376. 154.5 105.7 194.1 17.1 952.8 797.8 178.6]
  mean sample @target=80: [376. 154.5 105.7 194.1 17.1 952.8 797.8 178.6]
  IDENTICAL (target ignored): True
  ```
- `train()` builds `theta`/`x` and then `pass`es — it fits nothing — but still sets
  `is_trained = True`. After "training", `amortizer is None`.
- `evaluate_uncertainty()` returns `np.random.uniform(0.1, 0.9)` — pure noise.
- The module does `import tensorflow as tf` at top level with no guard, so **`import src.bayesian`
  fails outright** when TF isn't installed. Because `app.py` imports `BayesFlowExplorer` at module
  scope, **the entire Streamlit app fails to import** in any environment without TensorFlow — even
  though no real BayesFlow/TF computation is ever performed.

**Why this matters most:** `app.py` Tab 2 ("Amortized Performance Estimates") renders
`sample_posterior(target_str)` as a 3D surface captioned *"the probability distribution of mix
designs that could achieve it. Peaks indicate likely parameter combinations."* That surface is
target-independent Gaussian noise. The most prominent generative claim in the UI is
**actively misleading**, not merely incomplete. The technical report is admirably honest about other
limitations (§3.3 even flags the simulation gap) but never states that the amortizer is unimplemented.

**Fix path (incremental, no TF required to start):**
1. Make the TF/BayesFlow import optional (`try/except`, already done for `bf`, do the same for `tf`)
   so the app degrades gracefully instead of crashing.
2. Replace the noise sampler with an honest **rejection/importance sampler over the existing
   XGBoost model**: draw from the prior, keep samples where `|predict(θ) − target| < tol`. This is
   genuinely conditioned on the target, needs only the forward model, and is a correct (if slower)
   posterior. Label it as such.
3. Only then, if real-time inference is needed, train an actual amortized posterior and swap it in
   behind the same interface. Run the SBC diagnostic the report admits (§5.2) has never been run.
4. Until a real posterior exists, **relabel the UI** so users aren't told noise is a learned posterior.

### 1.2 `inverse_plan_mix` misses its targets and extrapolates out of distribution — HIGH

The heuristic inverse planner runs, but round-tripping its output through the forward model shows it
does not achieve the requested strength and is barely monotonic:

```
target=25 MPa -> cement=260 slag=0   water=117 w/c=0.45 | predicted=51.1 MPa
target=45 MPa -> cement=278 slag=142 water=97  w/c=0.35 | predicted=67.2 MPa
target=65 MPa -> cement=278 slag=272 water=97  w/c=0.35 | predicted=69.1 MPa
```

Problems:
- **No target tracking.** A request for 25 MPa yields a mix the model scores at 51 MPa; 45 and 65 MPa
  collapse to ~67–69 MPa. There is no feedback loop closing the planner against the forward model.
- **Water is computed against cement only**, ignoring SCM. For target=65 the binder is 278+272 slag
  but water is `0.35 × cement = 97`, giving a water/binder ratio of ~0.18 — physically unplaceable.
- **Out-of-distribution inputs.** UCI water ranges 121.8–247 kg/m³; the planner emits 97–117 kg/m³,
  below the training minimum. XGBoost then extrapolates into a region it never saw, which is exactly
  why the predictions overshoot. The planner should clamp to the data envelope (and the predictor
  should flag extrapolation).
- The carbon constraint is satisfied by construction but never **verified** against
  `calculate_embodied_carbon`, so "target_carbon_kg" is advisory at best.

**Fix:** close the loop — either (a) a small bounded optimizer (scipy `minimize`/`differential_evolution`)
that minimizes `|predict(m) − target|` subject to box constraints inside the data envelope, or (b) reuse
the existing GA/SA optimizers, which already do constrained search. The standalone heuristic adds risk
without adding capability the optimizers don't already provide.

---

## 2. Chemical models

### 2.1 `physics.py` is broken and dead — HIGH (latent)

`calculate_embodied_carbon` **does not exist as a function**. The intended body sits *after* the
`return total_cost` in `calculate_mix_cost`, as an unreachable docstring + dead code:

```
hasattr(physics, 'calculate_embodied_carbon') -> False
attrs: ['CARBON_FACTORS','UNIT_COSTS','calculate_mix_cost','estimate_curing_time', ...]
```

It happens not to bite today because nothing imports `physics` (app/notebook use
`chemistry_simple`), but the file is listed as a live component in the technical report
manifest (Appendix C) and duplicates `chemistry_simple`'s `CARBON_FACTORS`/`UNIT_COSTS` verbatim.

**Fix:** delete `physics.py` (its working half is a duplicate of `chemistry_simple`), or repair the
function and add a test. Keeping a broken, duplicated module invites a future caller to import the
missing symbol. Consolidate carbon/cost constants in **one** module.

### 2.2 Two carbon models disagree, and the report's claim about the direction is backwards — MEDIUM

Running both tiers on pure-OPC mixes:

```
cement=200: simple_total=194.2  adv_clinker_only=167.2
cement=350: simple_total=331.0  adv_clinker_only=292.6
cement=500: simple_total=467.8  adv_clinker_only=418.0
```

The simple (mass-factor) model is **higher** than the advanced (clinker calcination + fuel) model by
~12–16%. Yet the technical report (Recommendation §6.3) states *"The simple model underestimates
carbon for high-clinker cements by ~10–15%."* That is the wrong sign — and the two numbers aren't even
comparable, because `carbon_from_clinker` counts only cement (calcination + kiln fuel) while
`calculate_embodied_carbon` also sums aggregate/water/SP contributions. They measure different
system boundaries.

**Fix:** make the advanced tier a true superset (clinker chemistry **for cement** + the same
aggregate/water/admixture terms the simple tier uses) so a tier toggle changes *fidelity*, not
*scope*. Then correct the report's claim with the measured direction.

### 2.3 Advanced chemistry is reasonable but uncalibrated and partly disconnected — MEDIUM

- `bogue_calculation` runs and gives plausible phases (C₃S≈57%, C₂S≈18% for the default OPC). The
  report already concedes Bogue's known inaccuracy (§3.2) — fine as a screening estimate, but the
  hard clamps (`C3S ∈ [0,80]`, etc.) silently mask bad oxide inputs instead of warning.
- The **pozzolanic / hydration chemistry never reaches strength.** `analyze_mix` computes C-S-H,
  degree of hydration, and pozzolanic C-S-H, but none of it feeds the strength prediction — strength
  comes solely from XGBoost on raw masses. So the "molecular" tier is decorative with respect to the
  numbers users act on. Either wire C-S-H/maturity in as a forward feature (physics-informed), or be
  explicit that it's an explanatory side-panel, not part of the prediction.
- `inverse_plan_mix` and `carbon_from_clinker` use a hardcoded `clinker_factor=0.95` and ignore
  `data/oxide_compositions.json` (which already defines LC3 at 0.50, slag/ash at 0.0). The richest
  chemical lever in the dataset — clinker substitution — is left on the table.

### 2.4 `chemistry_simple` — works, minor notes — LOW

- Carbon/cost/curing all run and are dimensionally sane (350-cement mix: 343.7 kg CO₂/m³, $133/m³,
  9 days). Good as a fast screening tier.
- `estimate_curing_time` is a linear heuristic with no floor tie to actual strength and can produce
  odd values at extreme w/c; acceptable for screening, but it should be labeled heuristic in the UI
  (it currently reads as a metric next to model-backed strength).
- Transport carbon (`0.1 kg CO₂/tonne·km`) is a single global constant — fine as a placeholder,
  but it's one of the "carbon questions" the project brief explicitly wanted as a real input.

---

## 3. Priority recommendations

1. **Stop presenting noise as inference.** Guard the TF import, relabel Tab 2, and replace
   `sample_posterior` with target-conditioned rejection sampling over the XGBoost model. (Critical)
2. **Close the inverse loop.** Route `inverse_plan_mix` through a bounded optimizer constrained to the
   data envelope, or hand the job to the existing GA/SA. (High)
3. **Delete or fix `physics.py`** and de-duplicate carbon/cost constants into one module. (High)
4. **Unify the two carbon tiers** onto the same system boundary and correct the report's reversed
   claim. (Medium)
5. **Wire chemistry into prediction or label it explanatory**, and load `clinker_factor` from
   `oxide_compositions.json` instead of hardcoding 0.95. (Medium)

The forward model is a solid foundation. The generative half needs to either become real or be
labeled honestly before it reaches users — that is the single most important change.
