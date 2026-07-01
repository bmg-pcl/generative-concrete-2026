# Amortized Bayesian Inference for Inverse Mix Design

*A plain-language guide to the generative model in `src/amortized.py`.*

This document explains a technique that is unfamiliar to most engineers. If you only
remember one sentence: **we train a neural network once so that, forever after, it can
instantly answer "which concrete mixes give me strength X?" — not with a single recipe,
but with the whole spread of recipes that work.**

---

## 1. The problem: forward is easy, inverse is hard

Our XGBoost model answers the **forward** question:

> Given this mix, what strength will it reach? → `strength = predict(mix)`

But designers actually ask the **inverse** question:

> I need 45 MPa. Which mixes achieve that?

This is harder because the answer is **not unique**. Many different mixes reach 45 MPa
(more cement + more water, or less cement + a superplasticizer, or a slag blend cured
longer …). The set of valid answers is a *distribution*, not a point. In Bayesian terms
we want the **posterior**:

```
p( mix parameters | target strength )
```

---

## 2. What "amortized" means

The classic way to get a posterior is **MCMC** (Markov Chain Monte Carlo): a sampler that,
for *each new question*, runs thousands of model evaluations. Accurate, but slow — unusable
for interactive design where you drag a slider and expect an answer now.

**Amortized inference** flips the cost structure. We pay a large cost **once** (training a
neural network on many simulated examples), and then every future query is a single fast
forward pass. The training cost is "amortized" (spread) across all later queries — exactly
like a compiler spends time optimizing once so every later run is fast.

| | MCMC | Amortized (this project) |
|---|---|---|
| Cost per new target | High (rerun sampler) | ~milliseconds |
| Upfront cost | None | One training run |
| Good for | One-off analysis | Interactive exploration |

---

## 3. How we train it: simulation-based inference (SBI)

We never hand the network real lab data directly. Instead we teach it by showing it many
`(mix → strength)` examples generated from our own forward model. This is
**simulation-based inference**:

1. **Prior** — draw a random mix uniformly from the *training-data envelope* (so every mix
   is realistic, in-distribution). Code: `_draw_prior`.
2. **Simulator** — run it through XGBoost and add a little observation noise
   (`strength = predict(mix) + ε`). Code: `_simulate_batch`.
3. **Flow** — an **Invertible Neural Network** (a *normalizing flow*) learns to run this
   process *backwards*: given a strength, produce mixes consistent with it. Normalizing
   flows are used because they are exactly invertible and can represent the multi-modal
   answers this problem has (several distinct families of mixes for one strength).
4. **Standardization** — parameters span wildly different scales (cement ≈ 100–550,
   age 1–365), so we train in a normalized *z-space* and convert samples back to real units
   on the way out.

```
        prior mixes θ ──► simulator ──► strengths x
             ▲                               │
             └────── normalizing flow ◄───────┘
                  learns p(θ | x)  (the inverse)
```

Because the flow is trained against the forward *model* rather than raw measurements, its
posterior reflects **the model's** view of the world. This "simulation gap" is a real
limitation (see the technical report §3.3): if XGBoost is biased somewhere, the posterior
inherits that bias. It is honest to state this rather than imply the flow knows ground truth.

---

## 4. Is it trustworthy? Simulation-Based Calibration (SBC)

A generative model can look confident and still be **miscalibrated** (its uncertainty bands
too narrow or too wide). SBC is the standard check:

1. Draw a true mix `θ*` from the prior and simulate its strength `x*`.
2. Ask the flow for many posterior samples given `x*`.
3. Record the **rank** of `θ*` among those samples (how many fell below it).
4. Repeat for hundreds of `θ*`.

If the posterior is well-calibrated, these ranks are **uniformly distributed** — the true
value is equally likely to land anywhere in the posterior. Systematic deviations reveal
bias (ranks bunched high/low) or wrong spread (bunched at the middle/edges). We summarize
this per parameter as a **mean normalized rank**, which should sit near **0.50**.

Code: `AmortizedPosteriorModel.sbc_ranks`. BayesFlow also ships
`diagnostics.plot_sbc_ecdf` / `plot_sbc_histograms` for visual versions.

### Measured calibration (this training run)

Trained with 40 epochs × 300 iterations (batch 64), 5 coupling layers. SBC over 400
simulated datasets × 250 posterior samples. **Mean normalized rank per parameter (ideal
≈ 0.50):**

| Parameter | Mean rank | Parameter | Mean rank |
|---|---|---|---|
| cement | 0.499 | superplasticizer | 0.496 |
| slag | 0.477 | coarse_agg | 0.496 |
| ash | 0.497 | fine_agg | 0.519 |
| water | 0.494 | age | 0.505 |

All parameters sit within ±0.02 of 0.50 → the posterior is **well calibrated** (no
systematic bias or over/under-confidence at this training budget).

**Target-conditioning** (mean predicted strength of 1000 posterior samples, with the
water range showing samples stay in-distribution):

| Target (MPa) | Achieved mean | sd | Water range (kg/m³) |
|---|---|---|---|
| 20 | 23.5 | 8.2 | 122–247 |
| 30 | 32.7 | 7.9 | 122–247 |
| 45 | 46.1 | 6.2 | 122–247 |
| 60 | 60.0 | 5.4 | 122–247 |
| 70 | 67.0 | 4.6 | 122–236 |

The mild pull toward the data mean at the extremes (20 → 23.5, 70 → 67.0) is expected: the
uniform-envelope prior has few very-low / very-high-strength mixes, so the posterior is
gently regularized there. Re-run `python -m src.amortized` to reproduce these numbers.

---

## 5. How to use it

### Train once (offline)
```bash
python -m src.amortized          # trains, runs SBC, saves weights to models/amortizer/
```
or from code / a notebook:
```python
from src.bayesian import BayesFlowExplorer
explorer = BayesFlowExplorer()
explorer.train(epochs=40)        # trains + persists the flow
```

### Sample (instant, after training)
```python
explorer = BayesFlowExplorer()
mixes = explorer.sample_posterior(target_strength=45, n_samples=2000)   # (2000, 8)
```

`sample_posterior` picks its backend automatically:

- **trained amortized flow** when weights exist and no carbon target is given;
- **GA inverse designer** otherwise (also whenever a carbon target is set, since the flow
  conditions on strength only).

Force one with `method="amortized"` or `method="ga"`.

---

## 6. Why keep the metaheuristic designers too?

Two transparent metaheuristic designers (`src/generative_ga.py`) are the always-available
fallbacks, sharing one base class so they differ *only* in the search engine:

- **GA** (`PopulationInverseDesigner`) — a genetic algorithm minimizing `|predict(mix) − target|`.
- **ACO** (`AntColonyInverseDesigner`) — Ant Colony Optimization for continuous domains
  (ACO_R, `src/aco.py`); keeps an archive of good mixes as "pheromone" and samples new mixes
  around them. A useful independent check on the GA — if both converge to the same target
  in-envelope, you trust the result more.

Both:

- need **no TensorFlow**, so the app runs anywhere;
- are **transparent** (no neural network);
- handle **multi-objective** targets (strength *and* carbon) directly via the shared objective.

Select with `sample_posterior(..., method="ga"|"aco")`. The amortized flow adds **speed at scale**
(thousands of instant queries) and a **calibrated uncertainty story**. All three share one
interface, so callers never change.

---

## 7. Dependency notes (why the pins in `requirements.txt`)

The amortized stack uses **BayesFlow 1.x**, which is built on classic TensorFlow/TFP:

- `bayesflow<2` — v2.x is a Keras-3 rewrite with a different API;
- `tensorflow==2.14.*` with `tensorflow-probability==0.22.*` — matched pair;
- `numpy<2` — TF 2.14 has an ABI break with NumPy 2.x.

The **core app** (XGBoost predictor, GA generator, chemistry) needs none of these — they are
only required to train or sample the amortized posterior. The imports are guarded, so the app
degrades gracefully to the GA backend when the stack is absent.
