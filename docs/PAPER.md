# Conformalized Inverse Design of Concrete Mixtures: Coupling Two-Tier Cement Chemistry with Amortized and Metaheuristic Generative Models

**Preprint — describes the methods implemented in this repository (`src/`), with all
constants, calibration values, and benchmark figures taken from the committed code,
artifacts, and tests. Reproduction commands in §10.**

## Abstract

We present a system for the inverse design of concrete mixtures that couples a
gradient-boosted forward strength model with (i) a two-tier chemical analysis layer
spanning linear life-cycle factors through clinker-phase chemistry with explicit
production-route (kiln fuel, electricity, carbon capture) attribution, and (ii) a
family of interchangeable generative processes — a simulation-trained normalizing
flow (amortized posterior), two population metaheuristics (GA, ACO_R), and
evolutionary multi-objective optimization (NSGA-II/III) — all searching a common,
data-derived design envelope under shared physical-validity constraints.
Distinctively, uncertainty is *load-bearing* rather than decorative: a single
distributional model provides both the point prediction (its median) and a split
conformalized 90% interval (its outer quantiles; empirical coverage 0.888 at nominal
0.90) that cannot cross the point estimate by construction, an out-of-support score
with a data-calibrated threshold gates extrapolation, and a "robust" formulation
optimizes the conformal lower bound subject to an in-support constraint, so every
returned design carries a guaranteed-strength statement valid within the model's
evidential reach. We report a controlled comparison of the generative backends
(target tracking, design diversity, support retention, latency), simulation-based
calibration of the flow, and a coherence pathology of an earlier two-model
architecture — a point estimate crossing its own interval in sparse regions — that we
resolve by joint distributional estimation. An active-learning criterion closes the
loop by ranking the laboratory experiments that shrink the
model's blind spots fastest.

---

## 1. Introduction

Concrete mixture proportioning is a set-valued inverse problem: many mixtures
$\mathbf{x} \in \mathbb{R}^8$ realize a given compressive strength $s^\*$, and the
designer's real interest is the *set* — subsequently filtered by carbon, cost,
workability, and batchability. Two failure modes dominate naive ML treatments:

1. **Confident extrapolation.** A regressor queried by an optimizer will happily
   report high strength for mixtures far from its training support; the optimizer
   then converges exactly there.
2. **Point answers to set questions.** A single "optimal" mixture hides the
   trade-off structure and the (large) equivalence class of designs that meet spec.

Our design principle is that the generative machinery must be *subordinate to
calibrated uncertainty*: the same conformal interval and support score that the UI
displays are the quantities the search optimizes and the constraints it satisfies.
The chemical layer follows a parallel principle: every carbon figure carries its
provenance (supplier EPD, database record, parametric model, or user override), and
production-route differences that identical chemistry cannot express — carbon
capture, kiln electrification — are modeled explicitly with a Scope 1/2 split.

Contributions: (a) a conformalized, support-gated robust inverse-design formulation
shared across four generative backends; (b) a two-tier, same-boundary chemistry layer
with a parametric clinker-source model and EPD override semantics; (c) a controlled
empirical comparison of amortized vs metaheuristic generation; (d) a documented
mean/quantile crossing pathology with a proposed remedy.

## 2. Data and forward model

**Dataset.** The UCI Concrete Compressive Strength dataset (Yeh, 1998): $n=1030$
mixtures, features $\mathbf{x} = (\text{cement}, \text{slag}, \text{fly ash},
\text{water}, \text{superplasticizer}, \text{coarse agg.}, \text{fine agg.},
\text{age})$ in kg/m³ (age in days), response $s$ = compressive strength (MPa).
A calibration mechanism appends user laboratory rows as an overlay and retrains,
with per-retrain RMSE/R² logged to a metrics history.

**Forward model.** A single XGBoost multi-quantile model (800 trees, $\eta=0.05$,
depth 6, pinball objective for $\{0.05, 0.5, 0.95\}$) on an 80/20 split; its median
is the point predictor $\hat{f}(\mathbf{x})$ (test RMSE 4.97 MPa, $R^2 = 0.90$) and
its outer quantiles are the interval (§3.1). Point estimate and interval are thus one
distribution — the coherence property analyzed in §8.3.

**Design envelope.** All generative processes search the axis-aligned box
$\mathcal{B} = \prod_j [\min_i x_{ij}, \max_i x_{ij}]$ spanned by the training data
— a deliberately conservative feasible region that prevents the optimizer from
leaving the data by construction (support gating within $\mathcal{B}$ is handled
separately, §3.2).

## 3. Calibrated uncertainty

### 3.1 Conformalized quantile regression (CQR)

The forward model's outer quantiles, rearrangement-sorted per row so
$\hat{q}_{0.05} \le \hat{q}_{0.5} \le \hat{q}_{0.95}$, supply the raw interval.
Following Romano, Patterson & Candès (2019), the 80% training split is divided into a
fit fold (75%, on which the model is trained) and a disjoint calibration fold; on the
calibration fold we compute symmetric conformity scores

$$E_i = \max\big(\hat{q}_{0.05}(\mathbf{x}_i) - y_i,\; y_i - \hat{q}_{0.95}(\mathbf{x}_i)\big),$$

and the correction $\hat{Q} = E_{(\lceil (n_{cal}+1)\,0.9 \rceil)}$ — the finite-sample
90% empirical quantile. The reported interval is
$[\hat{q}_{0.05} - \hat{Q},\; \hat{q}_{0.95} + \hat{Q}]$. With the committed
artifacts, $\hat{Q} = 3.48$ MPa and **measured coverage on the held-out test fold is
0.888** against the nominal 0.90 — the central claim the UI makes ("90% interval")
is verified by a regression test, not asserted. Because the point prediction is the
same model's median (§8.3), the interval contains it identically.

### 3.2 Out-of-support (novelty) score

Trust is quantified by a $k$-NN distance ratio in standardized feature space:
$\nu(\mathbf{x}) = d_k(\mathbf{x}) / \tilde{d}_k$, where $d_k$ is the distance to the
$k$-th nearest training point ($k=10$) and $\tilde{d}_k$ the median in-sample value
(here 1.138). The in-support threshold is *data-calibrated*: $\tau$ = the 95th
percentile of $\nu$ over the held-out test fold ($\tau = 1.96$ for the committed
artifacts) — i.e., genuine unseen in-distribution mixtures define "normal". Mixtures
with $\nu > \tau$ are flagged as extrapolation in every surface (UI, tickets, CLI)
and are penalized or excluded by the robust search (§6.4).

## 4. Chemical analysis layer

### 4.1 Tier 1 — linear constituent accounting

Embodied carbon on an A1–A3 (cradle-to-gate) boundary plus an A4 transport term:

$$C(\mathbf{x}) = \sum_j m_j\, e_j \;+\; \frac{\sum_j m_j}{1000}\, d_t \cdot 0.1,$$

with $e_j$ the per-material emission factor (kg CO₂e/kg) and $d_t$ transport km at
0.1 kg CO₂/t·km. Factors are **provenance-tagged records** in a material registry
(value, unit, boundary, region, vintage, source ∈ {epd, database, parametric,
placeholder}, reference, uncertainty), resolved per material in the order *project
override > supplier EPD > database record > parametric model > placeholder*; the
resolved source is disclosed per material on every exported mix ticket. Cost is the
analogous linear form. The registry is pluggable: a new admixture is a JSON record,
not a code change.

### 4.2 Tier 2 — clinker chemistry

The advanced tier replaces **only the cement term** of Tier 1 (identical non-cement
terms and boundary), so the tiers are directly comparable — a reconciliation
invariant enforced by tests: the per-source breakdown sums exactly to the single
carbon figure in both tiers.

**Clinker phases.** Bogue (1929) estimation from oxide mass fractions:
$C_3S = 4.071\,\text{CaO} - 7.600\,\text{SiO}_2 - 6.718\,\text{Al}_2\text{O}_3 -
1.430\,\text{Fe}_2\text{O}_3 - 2.852\,\text{SO}_3$, with the companion equations for
$C_2S$, $C_3A$, $C_4AF$, clamped to physical ranges. Oxide compositions (OPC, GGBS,
class-F/C fly ash; per-cement clinker factors, e.g. OPC 0.95, LC3 0.50) are data,
not code.

**Hydration.** A Parrot–Killoh-style kinetic surrogate: ultimate degree
$\alpha_u = \min(1, (w/c)/0.38)$; time evolution
$\alpha(t) = \alpha_u\,(1 - e^{-k t^{0.6}})$ with $k = 0.15\,(C_3S/60)$; C-S-H and
portlandite (CH) from simplified stoichiometry; pozzolanic consumption of CH with
reactivity-class kinetics $\text{extent} = \min(1, \kappa \ln(1 + t/7))$
($\kappa$: slag 0.8, class-C ash 0.5, class-F ash 0.3). These support the analysis
report; they do not feed the strength prediction, which remains data-driven.

**Clinker carbon with production-route attribution.** Per kg clinker:

$$
\begin{aligned}
\text{Scope 1} &= (\underbrace{0.53}_{\text{calcination}} + \underbrace{EF_{\text{fuel}}}_{\text{combustion}})\,(1 - r_c),\\
\text{Scope 2} &= \big(\underbrace{\kappa_{\text{kiln}}}_{\text{electrified heat}} + \underbrace{(1000)^{-1} r_c\,(0.53 + EF_{\text{fuel}})\, \epsilon_c}_{\text{capture energy}}\big)\; EF_{\text{elec}},
\end{aligned}
$$

where $r_c$ is the capture rate applied to the stack (process + combustion, never to
the capture energy itself), $\epsilon_c$ the capture electricity demand (default 150
kWh/tCO₂, heat-integrated), $\kappa_{\text{kiln}} = 0.95$ kWh/kg for a fully
electrified kiln (≈3.4 GJ/t) else 0, $EF_{\text{fuel}} \in$ {coal 0.40, petcoke 0.42,
natural gas 0.28, waste-derived 0.15, biomass 0.05, electric 0} and $EF_{\text{elec}}$
a regional grid/hydro/nuclear/PPA factor. Computed exemplars (kg CO₂/kg clinker,
test-pinned at ±10%): coal kiln 0.93; gas kiln 0.81; gas + 90% capture on EU grid
0.11 vs on hydro 0.08; electrified hydro kiln 0.53 (irreducible calcination);
electrified hydro + 90% capture 0.06. The decomposition makes the Scope 1→2 migration
of electrification, and the survival of the capture-energy penalty, explicit — and a
supplier EPD, when attached, overrides the parametric model entirely (measured beats
modeled). Without a source descriptor the legacy default $(0.53 + 0.35)$ per kg
clinker is bit-identical, preserving backward comparability.

### 4.3 Derived mechanical/constructability quantities

Tensile strength via the Eurocode 2 correlation
$f_{ctm} = 0.30 f_c^{2/3}$ ($f_c \le 50$), $2.12 \ln(1 + (f_c+8)/10)$ otherwise —
labeled as a correlation, unvalidated once non-dataset admixtures are active. Curing
time and workability are transparent heuristics (flagged as such): curing from w/b
and SCM ratios; workability flags at w/b < 0.35 without plasticizer or w/b > 0.65.

## 5. Physical-validity constraints

The box envelope does not imply batchability. We enforce absolute-volume closure

$$V(\mathbf{x}) = \sum_j \frac{m_j}{\rho_j} + a \;\in\; [1 - \delta,\, 1 + \delta]\ \text{m}^3,$$

with air fraction $a = 0.02$ and tolerance $\delta = 0.05$ **calibrated on the data**
(95th percentile of $|V-1|$ over the 1030 UCI rows is 0.029; 100% fall under 0.05),
so every real mixture is admitted with margin for density assumptions. Violations are
handled twice: as a penalty inside every generative objective (weight 200 MPa per m³
of excess imbalance — designs are found balanced *and* on-target natively, avoiding
the strength shift a post-hoc repair induces), and by an aggregate-rescaling repair
applied to sampled clouds. Age is treated as a **condition, not a design variable**:
when a design age is fixed, the age dimension's bounds degenerate to a point, so the
optimizer cannot meet a strength target by prescribing a longer cure — a distinct
mechanism from any penalty, and immune to weight tuning.

## 6. Generative processes

All backends expose one interface — sample a cloud
$\{\mathbf{x}_i\} \sim p(\mathbf{x} \mid s^\*)$, optionally with a carbon budget, a
fixed age, and the robust flag — and are interchangeable behind a single dispatcher.

### 6.1 Shared objective (metaheuristics)

$$
\mathcal{L}(\mathbf{x}) = \big|\, \hat{s}(\mathbf{x}) - s^\* \big|
\;+\; \max(0,\, C(\mathbf{x}) - C^\*)
\;+\; 200 \max(0,\, |V(\mathbf{x})-1| - \delta)
\;+\; 10 \max(0,\, \nu(\mathbf{x}) - \tau)\,[\text{robust}],
$$

where $\hat{s} = \hat{f}$ (mean) normally and $\hat{s} = \hat{q}_{0.05} - \hat{Q}$
(conformal lower bound) in robust mode — the search then returns designs whose
*guaranteed* strength meets the target.

### 6.2 Genetic algorithm

Real-coded GA over $\mathcal{B}$: binary tournament selection, single-point crossover
(rate 0.8), uniform-resample mutation (rate 0.1), population 80, 40 generations,
elitist best tracking. The generative output is not the incumbent but the **elite
quarter of the final population**, expanded to $n$ samples by Gaussian jitter (σ = 2%
of each parameter range), clipped to $\mathcal{B}$, volume-repaired, and re-pinned to
the fixed age — a transparent, population-based stand-in for a posterior.

### 6.3 Ant colony optimization for continuous domains (ACO_R)

Socha & Dorigo (2008): the pheromone structure is a rank-sorted archive of $k=20$
solutions; each of 40 ants selects an archived guide with rank weights
$w_\ell \propto \exp(-(\ell-1)^2 / 2(qk)^2)$, $q = 0.2$, and samples each coordinate
from a Gaussian centred on the guide with spread $\xi = 0.85$ times the mean absolute
deviation of the archive in that coordinate — tight where the archive agrees,
exploratory where it does not. Identical objective, cloud expansion, and constraints
as the GA; it serves as an independent metaheuristic control.

### 6.4 Amortized posterior (normalizing flow)

Simulation-based inference (BayesFlow; Radev et al., 2020): prior
$\mathbf{x} \sim \mathcal{U}(\mathcal{B})$; simulator
$s = \hat{f}(\mathbf{x}) + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, 2^2)$ MPa;
a conditional invertible network (5 coupling layers) trained online (40 epochs × 250
iterations, batch 64) in a standardized $z$-space to approximate
$p(\mathbf{x}_{\setminus \text{age}} \mid s, \text{age})$ — the seven non-age
parameters conditioned on strength **and design age** (§R7.2). Age is a design
*condition*, not a sampled latent: it is drawn from the prior during training (so it
is marginalized correctly) and supplied at inference, so a fixed design age — the
tool's *default* setting — is served by the flow in one forward pass rather than
routed to a metaheuristic. A query with age unspecified draws age from the prior per
sample, recovering the marginal $p(\mathbf{x} \mid s)$. **Calibration:**
simulation-based calibration (Talts et al., 2018) rank statistics over the seven
sampled parameters (300 prior datasets × 250 posterior draws) span the whole
(strength × age) grid by construction, since both conditions are prior-drawn;
well-calibrated dimensions have uniform ranks (normalized mean ≈ 0.5). **Scope
guards:** only carbon-budget queries route to the metaheuristics (the flow does not
condition on carbon); two fingerprints stored with the weights — the dataset row
count and a content hash of the forward model — plus a conditioning-schema tag detect
staleness (recalibration, a changed forward model, or an incompatible schema) and
demote the flow to the GA fallback rather than serving silently wrong posteriors. The
trained flow inherits the simulator's view of the world — the *simulation gap*: its
posterior is faithful to $\hat{f}$, not to the laboratory.

### 6.5 Multi-objective front (NSGA-II/III)

Where §6.1–6.4 answer a conditional query, the trade-off question is posed directly
(pymoo): minimize $(-\hat{s}(\mathbf{x}),\, C(\mathbf{x}),\, \text{cost}(\mathbf{x}))$
over $\mathcal{B}$ subject to $|V(\mathbf{x})-1| \le \delta$ (always) and
$\nu(\mathbf{x}) \le \tau$ (robust) as inequality constraints — so the returned front
is batchable and in-support *by construction*, not by post-filtering. In robust mode
the strength objective is again the conformal lower bound. NSGA-III uses Das–Dennis
reference directions for front coverage. The amortized posterior composes with NSGA
as a **warm start**: seeding the initial population with target-conditioned samples
accelerates convergence and keeps early generations in-distribution. Returned fronts
are verified non-dominated by an independent $O(n^2)$ dominance check in the test
suite.

## 7. Active learning and the calibration loop

The system proposes its own experiments. Candidates are drawn from the current
generative backend at the strength of interest and ranked by

$$\text{merit}(\mathbf{x}) = \frac{\tfrac{1}{2}\big(\text{UB}(\mathbf{x}) - \text{LB}(\mathbf{x})\big)\cdot \min(\nu(\mathbf{x}), 3)}{1 + |\hat{f}(\mathbf{x}) - s^\*|},$$

i.e. wide calibrated interval × (capped) novelty × proximity to the target — the
tests that most reduce uncertainty *where the user is designing*. Executed tests
re-enter through the calibration overlay; retraining refreshes the joint
distributional model, conformal correction, and support set together, logs RMSE/R² to
the metrics history, and (by dataset-row and forward-model content-hash fingerprints)
invalidates the amortized flow until retrained.

## 8. Empirical evaluation

### 8.1 Backend comparison

Clouds of 1500 samples per (backend, target); mean |prediction − target| (MAE),
mean per-parameter standard deviation (spread), fraction in-support, wall-clock
(single-threaded CPU; non-robust mode; reproduce via §10):

| backend | target (MPa) | MAE (MPa) | spread | in-support % | time (s) |
|---|---|---|---|---|---|
| flow | 25 | 5.98 | 71.6 | 16.7 | 0.22 |
| flow | 45 | 3.99 | 77.7 | 5.1 | 0.15 |
| flow | 65 | 3.96 | 70.0 | 2.2 | 0.15 |
| GA | 25 | 3.99 | 28.3 | 1.7 | 3.85 |
| GA | 45 | 1.08 | 41.5 | 0.0 | 4.18 |
| GA | 65 | 0.98 | 38.5 | 0.0 | 3.86 |
| ACO_R | 25 | 0.63 | 7.1 | 0.0 | 2.06 |
| ACO_R | 45 | 1.36 | 19.5 | 0.0 | 2.01 |
| ACO_R | 65 | 0.70 | 19.9 | 0.7 | 2.07 |

(Regenerated against the joint distributional model and the flow retrained against
it; the flow's target error improved ~15–20% now that it inverts the median it is
scored on.) The structure is the expected accuracy/diversity/latency triangle: the
flow is an order of magnitude faster per query and the most diverse, at 3–6× the
target error (its cloud is a genuine posterior spread under observation noise σ=2, not
an optimizer's elite); the metaheuristics track the target tightly with moderated
diversity. The uniformly low in-support fractions are informative: an unconstrained
match-the-target search gravitates to sparse regions — which is precisely the
argument for the robust formulation (§6.4 gates and §6.1 penalties), whose in-support
guarantee is exercised by the test suite rather than this benchmark. NSGA-II maps a
60-point front in 0.34 s at pop 60 × 30 generations.

### 8.2 Calibration results

Conformal 90% interval: empirical coverage **0.888** (test fold), correction
$\hat{Q}=3.48$ MPa; zero point-in-interval violations over the test fold and a
10⁴-point envelope sweep (§8.3). Novelty threshold $\tau = 1.96$ (95th percentile of
held-out novelty). SBC normalized mean ranks for the retrained flow are 0.486–0.532
across the eight dimensions (ideal 0.5). Clinker-source exemplars in §4.2 are pinned
by tests at ±10%.

### 8.3 A resolved pathology: mean/quantile crossing

An earlier two-model architecture trained the point predictor (squared-error XGBoost)
and the interval (quantile XGBoost) *separately*, so sparse regions admitted
$\hat{f}(\mathbf{x}) < \hat{q}_{0.05}(\mathbf{x}) - \hat{Q}$ — a point estimate below
its own calibrated lower bound. We observed it live: a robust design at target 45 MPa
reported lower bound 45.1 with point prediction 39.1. Interval coverage was unaffected
(a property of the quantile model plus conformal correction), but the display was
incoherent and a lower-bound objective could be optimized into regions where the two
models disagreed.

We resolved it structurally rather than cosmetically. The point predictor and the
interval now come from **one** model: a single multi-quantile XGBoost whose median is
the point prediction and whose $(0.05, 0.95)$ quantiles — after per-row rearrangement
sorting (Chernozhukov, Fernández-Val & Galichon, 2010) and the same conformal
correction — are the interval. Because the median is one of the sorted quantiles,
$\hat{q}_{0.05} - \hat{Q} \le \hat{q}_{0.5} \le \hat{q}_{0.95} + \hat{Q}$ holds
identically ($\hat{Q} \ge 0$): **crossing is impossible by construction**, verified by
zero violations over the held-out test fold and a 10⁴-point uniform sweep of the design
envelope (regression-gated). The cost is a modest point-accuracy trade — split-conformal
validity requires the single deployed model to hold out a calibration fold, so its
median is fit on 75% of the training split where the old mean model used 100%: test
RMSE rises 4.65 → 4.97 MPa (+6.8%; median-MAE 3.4 MPa) and coverage holds at 0.888.
We consider guaranteed coherence, and a single distribution every downstream property
model can inherit, worth ~0.3 MPa of RMSE. The amortized flow, which conditions on the
forward model's outputs, is retrained against the new median and its staleness guard
now includes a content hash of the forward model so any future change demotes a stale
flow to the metaheuristic fallback.

## 9. Limitations

(i) Single-source data: one 1998 laboratory corpus; the calibration loop is the
mitigation, not a solved problem. (ii) The envelope prior is axis-aligned — it admits
corner combinations never jointly observed; the novelty gate, not the box, carries
that burden. (iii) Tier-2 hydration is a teaching-grade surrogate, not a
thermodynamic model (no Parrot–Killoh phase-resolved kinetics, no GEMS-style
speciation); it deliberately does not feed strength. (iv) The flow's simulation gap
(§6.4): it now conditions on (strength, age) but still not on carbon, so carbon-budget
queries route to the metaheuristics. (v) Derived tensile/curing/workability are correlations
and heuristics, labeled as such. (vi) Emission factors are indicative defaults unless
an EPD is attached; accounting conventions for waste/biomass kiln fuels vary by
jurisdiction. (vii) Fixed-seed metaheuristic benchmarks are order-of-magnitude
comparisons, machine- and seed-dependent.

## 10. Reproducibility

All artifacts (models, conformal/support constants, flow weights + normalization,
dataset) are committed and regenerated together:

```bash
python -m src.models                 # joint quantile model, CQR, support set
python -m src.amortized              # flow training + SBC report
python -m scripts.benchmark_backends # §8.1 table -> docs/BENCHMARKS.md
python -m pytest                     # all §3-§6 claims are test gates (100+ tests)
python -m src.cli design --target 45 --backend ga --age 28   # headless query
```

Test gates corresponding to claims in this paper: conformal coverage;
novelty-threshold calibration; robust fronts in-support; fronts non-dominated and
volume-balanced; fixed-age invariance under jitter/repair; carbon-breakdown
reconciliation in both tiers with and without a clinker source; the clinker exemplar
table; EPD resolution order and ticket provenance; benchmark plumbing.

## References

1. Yeh, I.-C. (1998). Modeling of strength of high-performance concrete using
   artificial neural networks. *Cement and Concrete Research* 28(12), 1797–1808.
2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
   *Proc. 22nd ACM SIGKDD*, 785–794.
3. Romano, Y., Patterson, E., & Candès, E. (2019). Conformalized quantile
   regression. *NeurIPS 32*.
4. Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random
   World*. Springer.
5. Radev, S. T., Mertens, U. K., Voss, A., Ardizzone, L., & Köthe, U. (2020).
   BayesFlow: Learning complex stochastic models with invertible neural networks.
   *IEEE TNNLS* 33(4), 1452–1466.
6. Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A. (2018).
   Validating Bayesian inference algorithms with simulation-based calibration.
   *arXiv:1804.06788*.
7. Papamakarios, G., Nalisnick, E., Rezende, D. J., Mohamed, S., &
   Lakshminarayanan, B. (2021). Normalizing flows for probabilistic modeling and
   inference. *JMLR* 22(57), 1–64.
8. Cranmer, K., Brehmer, J., & Louppe, G. (2020). The frontier of simulation-based
   inference. *PNAS* 117(48), 30055–30062.
9. Socha, K., & Dorigo, M. (2008). Ant colony optimization for continuous domains.
   *European Journal of Operational Research* 185(3), 1155–1173.
10. Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist
    multiobjective genetic algorithm: NSGA-II. *IEEE Trans. Evol. Comput.* 6(2).
11. Deb, K., & Jain, H. (2014). An evolutionary many-objective optimization
    algorithm using reference-point-based nondominated sorting: NSGA-III.
    *IEEE Trans. Evol. Comput.* 18(4).
12. Bogue, R. H. (1929). Calculation of the compounds in Portland cement.
    *Ind. Eng. Chem. Anal. Ed.* 1(4), 192–197.
13. Parrot, L. J., & Killoh, D. C. (1984). Prediction of cement hydration.
    *Br. Ceram. Proc.* 35, 41–53.
14. EN 1992-1-1 (Eurocode 2). Design of concrete structures — tensile strength
    correlation $f_{ctm}(f_{ck})$.
15. Hammond, G., & Jones, C. *Inventory of Carbon & Energy (ICE)* v3, Univ. of Bath;
    WBCSD/CSI (2013), *The Cement CO2 and Energy Protocol*.
16. Blank, J., & Deb, K. (2020). pymoo: Multi-objective optimization in Python.
    *IEEE Access* 8, 89497–89509.
