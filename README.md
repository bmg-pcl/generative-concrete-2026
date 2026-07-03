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

Headless (no browser) — run a particular configuration from the shell:

```bash
python -m src.cli predict --mix mix.json --config gmd_session.json --ticket out.csv
python -m src.cli design  --target 45 --backend ga --age 28
python -m src.cli pareto  --algorithm nsga2 --out front.csv
python -m src.cli predict --mix mix.json --epd supplier_epds.json   # plug in EPD factors
```

The `--config` file is the app's session export (one schema everywhere); `--epd` attaches
supplier EPD values that override the database defaults and are disclosed, per material,
in the ticket's provenance rows.

Training the amortized flow is optional: without it, inverse design uses the transparent
GA/ACO backends. See [`docs/AMORTIZED_INFERENCE.md`](docs/AMORTIZED_INFERENCE.md) for a
plain-language explanation of amortized inference.

**Adding a material** (SCM, admixture, fiber) is a JSON edit, not a code change: append a
record to `data/materials.json` — density, dosage bounds, cost, and at least one
provenance-tagged carbon record — and it appears in the UI and every carbon/cost path.
See [`docs/specs/R6-materials-platform.md`](docs/specs/R6-materials-platform.md) §2.2.

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
from `data/oxide_compositions.json`, e.g. OPC 0.95 vs LC3 0.50). The UI toggle changes fidelity,
not scope. Emission factors (ICE database / WBCSD-CSI defaults), **transport distance**, and
**clinker/cement source** are all editable in the Config tab and apply across every tab.

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
pytest                       # config in pytest.ini (pythonpath=., testpaths=tests)
ruff check .                 # lint (config in ruff.toml)
pytest --cov=src --cov-fail-under=60   # coverage floor as run in CI
```

The suite covers the chemistry tiers, the generative designers (target-tracking, in-envelope
sampling), the exotic-admixture switch, the UI logic, and the amortized flow (SBC calibration,
save/load). The `ui/` tab renderers and `app.py` are exercised end-to-end by an AppTest smoke
test (`tests/test_app_smoke.py`), which also asserts slider→metric value propagation. The heavy
TensorFlow/BayesFlow tests auto-skip when that stack is absent, so the core suite runs anywhere.
CI (`.github/workflows/tests.yml`) runs ruff, then the core suite with a coverage floor (60%),
on every push and pull request. CI omits the heavy TF/BayesFlow stack, so the amortized-flow
tests skip and CI coverage (~63%) runs below a full local run (~69%); the floor sits below CI's
number and can be ratcheted up over time.

### Committed binary artifacts

The trained model and its support data are committed so CI and a fresh clone need no download or
training step: `models/strength_model.json`, `models/strength_quantiles.json`,
`models/support.npz`, and (when trained) `models/amortizer/` (~5 MB total). Regenerate them with
`python -m src.models` (mean/quantile/support) and `python -m src.amortized` (the flow), and
commit them together so the forward model and the flow trained against it stay in sync. If the
repo outgrows this, move the artifacts to git-lfs or release assets (decision deferred; policy
stated here so it is a choice, not a drift). The runtime `models/metrics_history.json` is
gitignored — it is per-machine training history, not a shared artifact.

---

## Documentation

- [`docs/TUTORIAL.md`](docs/TUTORIAL.md) — guided tour of the app with screenshots, in workflow order
- [`docs/WORKFLOW.md`](docs/WORKFLOW.md) — the end-to-end workflow: steps, decision points, and loops in order
- [`docs/BUSINESS_REPORT.md`](docs/BUSINESS_REPORT.md) — economics of low-emission concrete, market fragmentation (spec regimes by jurisdiction), Scope 1/2/3, and where this tool fits
- [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md) — the framework, decisions, and limitations
- [`docs/AMORTIZED_INFERENCE.md`](docs/AMORTIZED_INFERENCE.md) — what amortized inference is and how it is trained/calibrated
- [`docs/MODEL_CRITIQUE.md`](docs/MODEL_CRITIQUE.md) — the original point-in-time model critique
- [`docs/specs/`](docs/specs/README.md) — roadmap specs (R1 trustworthy numbers, R2 physical validity, R3 dormant value, R4 engineering)
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
