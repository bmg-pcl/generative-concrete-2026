# Generative Mix Design: A Computational Framework for Concrete Formulation

**Technical Report v2.0**  
*January 2026*

---

## Preface: On the Nature of This Problem

The design of concrete mixtures is, at its core, an *inverse problem*. The engineer does not ask "What strength will this mix achieve?" but rather "What mix will achieve 45 MPa at minimum cost?" The first question admits a straightforward forward computation; the second requires searching through an eight-dimensional parameter space constrained by chemistry, economics, and regulation.

It is a peculiar property of our discipline that we have spent decades refining forward models—from Abrams' law to neural networks—while the inverse problem remains largely unsolved. We default to lookup tables (ACI 211.1) or experienced intuition, neither of which scales well to novel materials or multi-objective optimization.

This report documents a framework that takes the inverse problem seriously. It does so through three mechanisms: (1) a forward predictor trained on empirical data, (2) a Bayesian inference engine that inverts the forward model, and (3) a chemistry layer that grounds predictions in physical reality rather than pure curve-fitting.

---

## 1. The Problem, Precisely Stated

### 1.1 The Forward Problem

Given a mix design vector **m** = (cement, slag, ash, water, superplasticizer, coarse_agg, fine_agg, age), predict the compressive strength f_c:

```
f_c = F(m) + ε
```

where F is the unknown true function mapping composition to strength, and ε represents experimental noise (typically 2-5 MPa for well-controlled lab tests).

This is a regression problem. Any sufficiently flexible function approximator—neural networks, gradient boosted trees, Gaussian processes—can achieve R² ≈ 0.90 on the UCI dataset. The choice of XGBoost is pragmatic: it trains quickly, handles tabular data well, and provides reasonable uncertainty estimates through ensemble variance.

**The uncomfortable truth**: R² = 0.92 means 8% of variance is unexplained. For a 40 MPa target, this translates to approximately ±4 MPa prediction intervals. The model is *useful*, not *authoritative*.

### 1.2 The Inverse Problem

Given a target strength f_c*, find all mix designs **m** such that:

```
P(F(m) ≈ f_c* | data) > threshold
```

This is fundamentally harder. The forward function F is many-to-one: infinitely many mixes can achieve 40 MPa. The space of valid solutions is not a point but a *manifold* in 8-dimensional space. Traditional optimization (gradient descent, grid search) finds a single point; what we need is a characterization of the entire manifold.

This is why we use Bayesian inference. Instead of finding *a* solution, we learn the *posterior distribution* P(**m** | f_c*, carbon*), which encodes all plausible mixes weighted by their probability and compliance with sustainability targets.

### 1.3 The Multi-Objective Complication: From 2D to N-Dimensional Chaos

The historical focus of concrete design was a two-dimensional trade-off: **Strength vs. Cost**. This was a manageable frontier, often governed by the simple inverse relationship between water-cement ratio and strength. One could "buy" strength with cement mass, and the resulting cost increase was linear and predictable.

The introduction of the sustainability mandate has shattered this simplicity. We have transitioned from a two-dimensional trade-off to an $N$-dimensional optimization problem where $N$ includes carbon footprint, durability metrics, and an explosion of "exotic" admixtures.

#### 1.3.1 The Explosion of Exotics
In the current design landscape, "cement" is no longer a monolithic ingredient. We must now account for:
- **Nano-Engineered Fillers**: Nano-silica and graphene oxide, which operate at concentrations two orders of magnitude lower than traditional additives but induce nonlinear changes in nucleation density.
- **Fibrous Reinforcement**: Steel, basalt, and polypropylene fibers, which provide post-crack ductility but introduce complex rheological constraints.
- **Complex Pozzolanic Interaction**: Calcined clays (metakaolin) and silica fume, which interact with the calcium hydroxide liberated during OPC hydration in a time-varying, competitive manner.

#### 1.3.2 Functional Objectives Beyond Strength
Furthermore, we are moving beyond strength as the sole performance metric. A globally optimal mix must now satisfy:
- **Embodied Carbon (LCA)**: Often the primary constraint in modern bids.
- **Workability (Slump/Flow)**: A hidden constraint. A low-carbon mix that cannot be pumped or placed is a failed design.
- **Durability (Permeability/Diffusion)**: Ensuring a 100-year service life shifts the optimization from 28-day performance to century-long chemical stability.

Each additional objective expands the search space exponentially. The "ideal" mix design is no longer a point on a curve, but a narrow bridge in a high-dimensional vacuum. Our framework uses metaheuristics specifically to survive this search without collapsing into local minima.

---

## 2. Architectural Decisions and Their Justifications

### 2.1 Why XGBoost, Not Neural Networks?

The UCI dataset contains 1,030 samples. Neural networks require thousands to millions of samples to avoid overfitting. Gradient boosted trees are native to tabular data: they handle mixed feature types, require minimal preprocessing, and provide built-in regularization through tree depth limits and shrinkage.

We considered:
- **Random Forests**: Comparable accuracy, but slower inference and no native gradient availability
- **SVR**: Requires careful kernel selection; less interpretable
- **Deep Neural Networks**: Overfitting risk; requires extensive hyperparameter tuning

XGBoost achieved the best cross-validated RMSE (4.65 MPa) with default hyperparameters. This is the hallmark of a well-suited algorithm: it works without heroic tuning.

### 2.2 Why Amortized Inference, Not MCMC?

Traditional Bayesian inference uses Markov Chain Monte Carlo (MCMC) to draw samples from the posterior. MCMC is asymptotically exact but slow: thousands of likelihood evaluations per query. For interactive design exploration, this is unacceptable.

Amortized inference trains a neural network to directly map observations to posterior samples. The training phase is expensive (hours), but inference is instantaneous (milliseconds). This is the critical trade-off: we pay upfront for real-time capability.

The specific architecture—Invertible Neural Networks / Normalizing Flows—is chosen because:
1. The inverse transform is exact (no approximation error in sampling)
2. The likelihood is tractable (enabling training via maximum likelihood)
3. Multimodality is naturally captured (unlike variational autoencoders)
4. **Active Experimental Design**: By targeting high-entropy regions of the posterior, we can suggest the **top-five most informative tests** to run next.

### 2.3 Why Two Chemistry Tiers?

The simple tier (linear carbon factors) is *wrong* but *useful*. It provides quick estimates suitable for early-stage design screening. An engineer who wants a rough carbon comparison between two mixes does not need Bogue calculations.

The advanced tier (molecular model) is *less wrong* but *slower*. It accounts for:
- Clinker phase composition (C₃S, C₂S, C₃A, C₄AF)
- Hydration kinetics (time-dependent strength development)
- Pozzolanic reactions (SCM contribution to long-term strength)
- Carbon from calcination vs. kiln fuel (more accurate than mass-based factors)

The user should toggle between tiers based on the decision context. Early screening: simple. Final validation: advanced.

---

## 3. On the Limitations of This Work

I list these not as disclaimers but as *research directions*. Each limitation implies a future improvement.

### 3.1 Data Limitations

**The UCI dataset is from Taiwan, 1998.** Materials science has advanced significantly:
- Modern superplasticizers (PCE-based) behave differently than naphthalene-based versions
- LC3 (Limestone Calcined Clay Cement) did not exist commercially in 1998
- Recycled aggregates are not represented

**Implication**: The model will underperform on novel materials. The calibration feature (uploading lab data) partially addresses this, but widespread adoption requires curating a modern, diverse dataset.

### 3.2 Modeling Limitations

**The forward model treats age as an input, not a process.** Real concrete strength evolves continuously; the model treats 7-day, 28-day, and 90-day as independent points. A physics-informed model would embed rate equations for hydration.

**The Bogue calculation is known to be inaccurate.** Industrial clinker does not reach thermodynamic equilibrium during cooling. Taylor's modified Bogue or Rietveld XRD refinement would be more accurate, but requires detailed inputs we do not have.

**Pozzolanic reaction is heuristic.** Real pozzolanic reaction follows Arrhenius temperature dependence and depends on SCM mineralogy (glass content, particle size distribution). Our model uses time-dependent scaling factors calibrated to literature averages.

### 3.3 Inference Limitations

**The amortizer is trained on XGBoost predictions, not real data.** This creates a simulation gap: the posterior is conditioned on the *model's* view of the world, not reality. If the model is biased, the posterior will be biased.

**No separation of aleatoric and epistemic uncertainty.** The current system cannot distinguish "this mix is inherently variable" (aleatoric) from "we don't have enough data about this mix" (epistemic). Deep ensembles or heteroscedastic models would be required.

---

### 4. Metaheuristic Optimization: Survival in High-Dimensional Space

As the number of design variables ($N$) and objectives ($M$) increases, the search space $S = \mathbb{R}^N$ grows exponentially—a phenomenon known as the **Curse of Dimensionality**. In concrete design, $N \ge 8$ and $M \ge 3$. Traditional gradient-based optimization fails because:
1. **Model Non-Convexity**: Neural networks and XGBoost do not possess a single global minimum.
2. **Disconnected Valid Regions**: Real-world constraints (e.g., aggregate grading curves) create non-contiguous islands of feasibility.
3. **Competing Gradients**: Improving Carbon often requires decreasing Strength; a local gradient-following algorithm will simply "chase its tail" at the Pareto frontier.

#### 4.1 Genetic Algorithms: Population-Level Survival
Our Genetic Algorithm implementation treats a population of 50–200 mixes as a collective search engine. By combining Simulated Binary Crossover (SBX) with high mutation rates (5%), the algorithm maintains "genetic diversity"—preventing the premature optimization of a high-strength mix that is economically unfeasible. The **Violin Plots** in the dashboard serve as a diagnostic to ensure the "gene pool" is exploring the full range of exotic admixtures before collapsing into an $N$-dimensional valley.

The genetic algorithm encodes each candidate mix as a chromosome (a vector of 8 floating-point genes). Evolution proceeds through:

1. **Selection**: Tournament selection with k=3 (random subset, winner survives)
2. **Crossover**: Simulated binary crossover (SBX) with distribution index η=20
3. **Mutation**: Polynomial mutation with probability 1/n per gene

The fitness function scalarizes multiple objectives:

```
where α = 0.05 and β = 0.5 by default. These coefficients encode priors about trade-off preferences; they are fully adjustable by the user in the dashboard.

**Multi-Objective Metric Tracking**: Unlike standard GA implementations that only report a single fitness score, our system decomposes the best solution at each generation into its constituent metrics (Strength, Carbon, Cost). This provides transparency into *how* the algorithm is making trade-offs—for instance, revealing if a fitness gain was achieved through higher strength or through a more aggressive carbon penalty.

**Convergence behavior**: GA is effective for global exploration but may converge prematurely to local optima. The population diversity typically decreases over generations; the heatmap visualization reveals this convergence.

#### 4.2 Simulated Annealing: The Thermodynamic Refinement
While GA explores broadly, Simulated Annealing (SA) allows for a single "walk" through the landscape with the ability to escape local traps. By accepting sub-optimal moves with a probability $P = \exp(-\Delta / T)$, SA can "jump" over the chemical barriers of a high-slag mix to find a globally optimal recipe that balances hydration kinetics with kiln fuel efficiency.

Simulated annealing maintains a single solution that performs a random walk through the search space. The key insight is the acceptance criterion:

```
P(accept) = exp(−ΔE / T)
```

At high temperature T, the system accepts worsening moves (escaping local optima). As T decreases, it becomes increasingly greedy. The cooling schedule `T_new = α × T_old` with α = 0.95 provides geometric cooling.

**When to use SA over GA**: SA is preferable when:
- Fine-tuning a known good solution is needed
- Computational budget is limited (SA evaluates fewer solutions per step)

#### 4.3 Amortized Suggestion: The Information-Theoretic Approach
Beyond metaheuristics, we employ the amortized posterior to guide laboratory work. The `suggest_tests` algorithm identifies mix designs that maximize a *merit score*—a balance between meeting performance targets and exploring high-uncertainty (high-variance) regions of the model.

This addresses the "empty spaces" in our knowledge by ensuring that new lab data provides the maximum possible gain in model accuracy (Active Learning).

---

## 5. Validation Philosophy

We distinguish three levels of validation:

### 5.1 Verification (Code Correctness)
Does the code do what the documentation says? This is tested via unit tests:
- Bogue calculation matches hand-calculated examples
- GA convergence on known test functions (Rastrigin, Rosenbrock)
- Data loading produces expected schema

### 5.2 Validation (Model Accuracy)
Does the model match reality? This requires external benchmarks:
- XGBoost: 80/20 train-test split, R² = 0.92 ✓
- Chemistry: Comparison against Taylor (1997) phase compositions ✓
- BayesFlow: Simulation-Based Calibration (SBC) diagnostics ✗ (not yet run)

### 5.3 Qualification (Fitness for Purpose)
Does the system solve the user's problem? This requires field deployment:
- Partner with a ready-mix plant to generate predictions
- Compare predictions to actual cylinder breaks
- Iterate on calibration

We are currently at validation level. Qualification remains future work.

---

## 6. Recommendations for Production Use

1. **Never use predictions for structural certification without laboratory validation.** The model is a design exploration tool, not a substitute for ASTM testing.

2. **Calibrate to your materials.** Upload at least 20-50 local lab results via the Calibration tab. Regional cement chemistry varies significantly.

3. **Use Advanced Chemistry for carbon accounting.** The simple model underestimates carbon for high-clinker cements by ~10-15%.

4. **Run GA for exploration, SA for refinement.** Use GA to identify promising regions of the Pareto front, then SA to fine-tune specific solutions.

5. **Document uncertainty.** When presenting results to stakeholders, include confidence intervals, not just point predictions.

---

## 7. Conclusion: The Value of Getting Less Wrong

Dijkstra famously remarked that "computer science is no more about computers than astronomy is about telescopes." Similarly, this project is not about concrete—it is about *inference under uncertainty*.

We cannot know the true strength of an unmixed concrete. We can only reason about what we expect given prior knowledge and observed data. The framework presented here makes that reasoning explicit, traceable, and improvable.

Every limitation documented above is a hypothesis about what would make the system more accurate. Every calibration upload is an experiment that refines those hypotheses. Over time, with disciplined data collection and honest assessment, the system converges toward truth.

This is the essence of engineering: not getting things right the first time, but having a principled method for getting *less wrong* over time.

---

## Appendix A: File Manifest

| Path | Purpose |
|------|---------|
| `app.py` | Streamlit user interface |
| `src/models.py` | XGBoost strength predictor |
| `src/chemistry_simple.py` | Tier 1 linear carbon model |
| `src/chemistry_advanced.py` | Tier 2 Bogue/hydration model |
| `src/bayesian.py` | BayesFlow amortized inference |
| `src/ga.py` | Genetic algorithm optimizer |
| `src/annealing.py` | Simulated annealing optimizer |
| `src/data_fetcher.py` | UCI dataset acquisition |
| `src/viz.py` | Visualization utilities |
| `data/oxide_compositions.json` | Default cement chemistries |
| `docs/TECHNICAL_REPORT.md` | This document |

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Amortized Inference** | Pre-training a neural network to perform Bayesian inference in constant time |
| **Bogue Calculation** | Classical method for estimating clinker phase composition from oxide analysis |
| **C-S-H** | Calcium Silicate Hydrate, the primary binding phase in hardened cement |
| **Normalizing Flow** | A neural network architecture that learns invertible transformations |
| **Pareto Frontier** | The set of solutions where no objective can be improved without worsening another |
| **Pozzolanic Reaction** | Secondary reaction between SCMs and calcium hydroxide to form additional C-S-H |
| **SCM** | Supplementary Cementitious Material (slag, fly ash, silica fume, etc.) |
| **Scalarization** | Combining multiple objectives into a single scalar fitness function |

---

*"The purpose of computing is insight, not numbers."* — Richard Hamming

*Document Version 2.0 | January 2026*
 
---

## Appendix C: Models and Technical Types

The table below lists the primary models, their implementation locations, and the technical type (ML/AI, chemical, or optimization algorithm).

| Model / Artifact | Path | Technical Type |
|---|---|---|
| `Strength Predictor` (XGBoost regressor) | `src/models.py` / `models/strength_model.json` | ML / Supervised regression (Gradient Boosted Trees — XGBoost) |
| `Predictive Mix Performance ` (Amortized inference) | `src/bayesian.py` | ML / Amortized Bayesian Inference (Normalizing Flows — generative/inference) |
| `Genetic Mix Optimizer` | `src/ga.py` | Optimization / Metaheuristic (Genetic Algorithm — population-based search) |
| `Simulated Annealing Optimizer` | `src/annealing.py` | Optimization / Metaheuristic (Simulated Annealing — thermodynamic-inspired search) |
| `Simple Accrual Chemistry` | `src/chemistry_simple.py` | Chemical / Heuristic linear constitutive model (carbon & cost estimates) |
| `Molecular Chemistry` | `src/chemistry_advanced.py` | Chemical / Molecular-level thermodynamic & kinetic model (Bogue, hydration, pozzolanic reactions) |
| `Physics Heuristics` | `src/physics.py` | Empirical / Heuristic physics and cost-carbon utilities |
| Strength model artifact | `models/strength_model.json` | ML artifact (serialized XGBoost model) |
| Oxide composition data | `data/oxide_compositions.json` | Chemical reference data (oxide compositions for common SCMs/clinkers) |

Notes:
- ML models are trained or used for prediction/inference (supervised regression, amortized posterior sampling).
- Chemical models implement domain physics or heuristics (forward simulations, Bogue calculation, hydration kinetics).
- Optimization algorithms are search strategies used to explore the inverse design space and produce candidate mixes.
