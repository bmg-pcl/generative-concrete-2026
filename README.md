# Generative Mix Design

A framework for concrete mix design that treats formulation as an **inverse problem**:
instead of only predicting the strength of a given mix, it generates realistic mixes that
hit a target strength while trading off embodied carbon and cost.

It combines a fast **XGBoost** forward predictor, three interchangeable **inverse-design
backends** (a trained amortized BayesFlow normalizing flow, plus transparent genetic-algorithm
and ant-colony designers), a two-tier **chemistry** model for carbon and cost, and
multi-objective **Pareto optimization** — all behind a Streamlit UI.

> Design-exploration tool, not a substitute for lab testing. Predictions must never be used
> for structural certification without physical validation (see the caveats below).

---

## Quickstart

```bash
git clone <repository-url>
cd generative-concrete-2026
pip install -r requirements.txt      # or: uv pip install -r requirements.txt
```

```bash
python -m src.data_fetcher --check   # fetch/verify the UCI concrete dataset
python -m src.amortized              # (optional) train the BayesFlow flow + run SBC
streamlit run app.py                 # launch the app
```

Training the amortized flow is optional: without it, inverse design uses the transparent
GA/ACO backends. See [`docs/AMORTIZED_INFERENCE.md`](docs/AMORTIZED_INFERENCE.md) for a
plain-language explanation of amortized inference.

---

## What it does

**Forward prediction.** An XGBoost regressor trained on the UCI concrete dataset (1,030 samples,
8 features) predicts 28-day-style compressive strength (RMSE ~4.65 MPa, R² ~0.92).

**Inverse design (generative).** Given a target strength, produce a spread of realistic mixes
that achieve it, all clamped to the training-data envelope. Three backends behind one interface:

| Backend | What it is | Needs TensorFlow |
|---|---|---|
| Amortized flow | Trained BayesFlow normalizing flow; instant, calibrated (SBC-checked) | yes |
| GA designer | Genetic algorithm minimizing \|predict − target\| | no |
| ACO designer | Ant colony optimization for continuous domains (ACO_R) | no |

**Chemistry (carbon & cost).** Two tiers on the same system boundary: a linear mass-factor
tier (fast screening) and a clinker-chemistry tier (Bogue phases, hydration, clinker factors
loaded from `data/oxide_compositions.json`, e.g. OPC 0.95 vs LC3 0.50). The UI toggle changes
fidelity, not scope.

**Multi-objective optimization.** Two paths: GA / simulated annealing search a *weighted*
objective (strength vs. carbon vs. cost), and **NSGA-II / NSGA-III** (via
[pymoo](https://pymoo.org)) map the *true* Pareto front across all three objectives with no
weights — and can **warm-start** from the inverse-design flow. See
[`docs/WORKFLOW.md`](docs/WORKFLOW.md) for when to use which.

**Calibration.** Upload lab results (CSV) to append to a local overlay and retrain on your
own materials.

---

## Architecture

```
data/Concrete_Data.xls ── data_fetcher ──> load_data()
                                             │
                              models.StrengthPredictor (XGBoost)  ── forward
                                             │
        ┌───────────────── inverse design (one interface) ─────────────────┐
        │  bayesian.BayesFlowExplorer.sample_posterior(method=…)            │
        │    auto → flow when trained, else GA                              │
        │    flow → amortized.AmortizedPosteriorModel (trained flow + SBC)  │
        │    ga   → generative_ga.PopulationInverseDesigner                 │
        │    aco  → generative_ga.AntColonyInverseDesigner (aco.py)         │
        └──────────────────────────────────────────────────────────────────┘
   chemistry_simple / chemistry_advanced  →  carbon & cost (two tiers)
   ga.py / annealing.py                   →  weighted (scalarized) search
   nsga.py (pymoo)                        →  NSGA-II / NSGA-III true Pareto front
   ui_logic.py                            →  pure, testable UI logic (metrics,
                                             fitness, recipe, Pareto front, validators)
   app.py                                 →  Streamlit UI (Compare / Inverse Design /
                                             Pareto / Calibration / Config / docs)
```

Core numeric logic lives in `src/` and is importable/testable without Streamlit; `ui_logic.py`
keeps a single carbon path, metrics path, and fitness path so every tab stays consistent.

---

## Testing

```bash
pytest        # config in pytest.ini (pythonpath=., testpaths=tests)
```

The suite covers the chemistry tiers, the generative designers (target-tracking, in-envelope
sampling), the exotic-admixture switch, the UI logic, and the amortized flow (SBC calibration,
save/load). The heavy TensorFlow/BayesFlow tests auto-skip when that stack is absent, so the
core suite runs anywhere. CI (`.github/workflows/tests.yml`) runs the core suite on every push
and pull request.

---

## Documentation

- [`docs/WORKFLOW.md`](docs/WORKFLOW.md) — the end-to-end workflow: steps, decision points, and loops in order
- [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md) — the framework, decisions, and limitations
- [`docs/AMORTIZED_INFERENCE.md`](docs/AMORTIZED_INFERENCE.md) — what amortized inference is and how it is trained/calibrated
- [`docs/MODEL_CRITIQUE.md`](docs/MODEL_CRITIQUE.md) — the original point-in-time model critique
- [`docs/FIX_PLAN.md`](docs/FIX_PLAN.md) / [`docs/UI_FIX_PLAN.md`](docs/UI_FIX_PLAN.md) — the sequenced fix plans

---

## Honest caveats

- **Simulation gap.** The amortized flow is trained against the XGBoost forward *model*, not raw
  lab data, so its posterior reflects the model's view of the world.
- **Exotic admixtures** (silica fume, fibers, nano-silica, …) affect cost and carbon by default;
  their effect on *strength* is an opt-in, **unvalidated** placeholder (the model has no exotics
  in its training data). Enable it in the Config tab only with that caveat in mind.
- **Dataset vintage.** The UCI data is from 1998 Taiwan; modern admixtures and regional materials
  are not represented. Use the Calibration tab to adapt the model to your materials.
- **Not for certification.** This is a design-exploration tool. Validate with physical testing
  (ASTM/EN) before any structural use.
