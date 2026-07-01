import numpy as np
import pandas as pd
import os

# TensorFlow / BayesFlow are heavy, optional dependencies. The current explorer
# does not require them at runtime (see Phase 3 in docs/FIX_PLAN.md), so we guard
# the imports to keep `import src.bayesian` — and therefore the whole app — working
# in environments without them installed.
try:
    import tensorflow as tf
except ImportError:
    tf = None
try:
    import bayesflow as bf
except ImportError:
    bf = None

from .data_fetcher import load_data
from .models import StrengthPredictor
from .chemistry_simple import calculate_embodied_carbon
from .generative_ga import PopulationInverseDesigner, AntColonyInverseDesigner, PARAM_NAMES

class BayesFlowExplorer:
    """
    Explores the inverse design space p(parameters | target strength).

    GENERATIVE BACKENDS (see docs/FIX_PLAN.md and docs/AMORTIZED_INFERENCE.md):
    `sample_posterior` has two interchangeable backends behind one interface:

      1. A trained **amortized BayesFlow posterior** (a normalizing flow in
         `src/amortized.py`) -- used automatically when trained weights exist and
         no carbon target is given (the flow conditions on strength only).
      2. A transparent **GA inverse designer** (`PopulationInverseDesigner`) --
         the always-available fallback; it needs no TensorFlow and also handles the
         carbon objective.

    Both are genuinely conditioned on the requested target (unlike the original
    placeholder, which returned an identical noise cloud for every query).
    """

    def __init__(self):
        self.predictor = StrengthPredictor()
        self.amortizer = None
        self.is_trained = False
        self.param_names = list(PARAM_NAMES)
        # Lazily-built backends (built on first use, sharing our predictor).
        self._designer = None
        self._aco_designer = None
        self._amortized = None  # None=untried, False=unavailable, else the model

    @property
    def designer(self) -> PopulationInverseDesigner:
        """The GA-based inverse designer, built on first use (shares our predictor)."""
        if self._designer is None:
            self._designer = PopulationInverseDesigner(predictor=self.predictor)
        return self._designer

    @property
    def aco_designer(self) -> AntColonyInverseDesigner:
        """The ACO-based inverse designer, built on first use (shares our predictor)."""
        if self._aco_designer is None:
            self._aco_designer = AntColonyInverseDesigner(predictor=self.predictor)
        return self._aco_designer

    @property
    def amortized(self):
        """
        The trained amortized BayesFlow posterior if available, else None.

        Available means: the pinned TF/BayesFlow stack is installed AND trained
        weights exist on disk (models/amortizer/). Result is cached; when it is
        None the callers transparently fall back to the GA designer.
        """
        if self._amortized is False:
            return None
        if self._amortized is None:
            self._amortized = self._load_amortized()
        return self._amortized or None

    def _load_amortized(self):
        try:
            from .amortized import AmortizedPosteriorModel
            if not AmortizedPosteriorModel.weights_exist():
                return False
            model = AmortizedPosteriorModel(predictor=self.predictor)
            if not model.load():
                return False
            self.is_trained = True
            return model
        except Exception as e:  # missing TF/BayesFlow, or load failure
            print(f"Amortized posterior unavailable ({type(e).__name__}: {e}); using GA designer.")
            return False

    def train(self, epochs=40, iterations_per_epoch=300, batch_size=64):
        """
        Train and persist the amortized BayesFlow posterior (see src/amortized.py).

        Requires the pinned TF/BayesFlow stack (requirements.txt). Trains a
        normalizing flow via simulation-based inference against the XGBoost forward
        model, saves the weights, and wires them in so `sample_posterior` uses the
        flow automatically. Returns the trained AmortizedPosteriorModel.
        """
        from .amortized import AmortizedPosteriorModel
        model = AmortizedPosteriorModel(predictor=self.predictor)
        model.train(epochs=epochs, iterations_per_epoch=iterations_per_epoch, batch_size=batch_size)
        model.save()
        self._amortized = model
        self.is_trained = True
        return model

    def sample_posterior(self, target_strength: float, carbon_target: float = None,
                         n_samples: int = 2000, method: str = "auto",
                         robust: bool = False, age: float = None) -> np.ndarray:
        """
        Draw mix designs conditioned on the target strength (and optional carbon target).

        Backend selection:
          * method="auto" (default): use the trained amortized flow when it exists
            AND no carbon target is given; otherwise the GA inverse designer.
          * method="amortized" / "flow": force the flow (errors if none is trained).
          * method="ga": force the GA designer.
          * method="aco": force the Ant Colony (ACO_R) designer.

        The flow conditions on strength only, so carbon targets always route to a
        metaheuristic designer (which bakes carbon into its objective).
        """
        if method in ("amortized", "flow") and self.amortized is None:
            raise RuntimeError(
                "No trained amortized posterior available. Train it first via "
                "BayesFlowExplorer.train() / `python -m src.amortized`, or use method='ga'."
            )
        if method == "aco":
            return self.aco_designer.sample(
                target_strength, n_samples=n_samples, carbon_target=carbon_target,
                robust=robust, age=age,
            )
        use_amortized = (
            method in ("auto", "amortized", "flow")
            and carbon_target is None
            and age is None            # the trained flow can't honor a fixed design age
            and self.amortized is not None
        )
        if use_amortized:
            # The trained flow can't be re-conditioned on the robust objective; robust
            # handling for the flow happens in recommend_recipe (filter/lower-bound match).
            # TODO(flow-age-conditioning): retrain the flow with (strength, age) conditioning
            # so a fixed age can use the flow; until then, pinning age routes to the GA.
            return self.amortized.sample(target_strength, n_samples=n_samples)
        return self.designer.sample(
            target_strength,
            n_samples=n_samples,
            carbon_target=carbon_target,
            robust=robust,
            age=age,
        )

    def suggest_tests(self, target_strength: float, carbon_target: float = None, n_tests: int = 5) -> pd.DataFrame:
        """
        Suggests the top-N tests to run based on a combination of target match and high uncertainty.
        This guides the user toward 'empty spaces' in the design manifold.
        """
        # 1. Broadly sample the posterior
        samples = self.sample_posterior(target_strength, carbon_target, n_samples=min(2000, 500 * n_tests))
        
        # 2. Score each sample based on uncertainty (predict_variance)
        results = []
        for s in samples:
            mix_dict = dict(zip(self.param_names, s))
            strength = self.predictor.predict(s)
            carbon = calculate_embodied_carbon(mix_dict)
            uncertainty = self.predictor.predict_variance(s)
            
            # Merit score: blends proximity to target with high uncertainty (exploration)
            strength_error = abs(strength - target_strength)
            # We want LOW strength error but HIGH uncertainty
            merit_score = uncertainty / (1.0 + strength_error)
            
            results.append({
                **mix_dict,
                "predicted_strength": strength,
                "embodied_carbon": carbon,
                "uncertainty_score": uncertainty,
                "merit_score": merit_score
            })
            
        df = pd.DataFrame(results)
        # Select the top-N diverse tests
        # Sorting by merit score but we could also use a diversity filter (e.g. K-Means)
        top_tests = df.sort_values("merit_score", ascending=False).head(n_tests)
        
        return top_tests

    def evaluate_uncertainty(self, mix_design: np.ndarray) -> float:
        """
        Quantifies how 'empty' (uncertain) the design space is around a mix.

        Backed by the predictor's heuristic variance estimate rather than a random
        number: mixes with extreme w/c ratios or high SCM replacement -- regions the
        1998 UCI data covers sparsely -- score higher. This is still a heuristic
        (a real system would use deep ensembles), but it is at least deterministic
        and grounded in the mix, not noise.
        """
        return float(self.predictor.predict_variance(np.asarray(mix_design, dtype=float)))

    def explain_empty_spaces(self) -> str:
        backend = "trained amortized normalizing flow" if self.amortized is not None \
            else "GA inverse designer (train the flow to enable the BayesFlow backend)"
        return (
            "### Amortized Bayesian Inference\n"
            f"**Active backend:** {backend}.\n\n"
            "We learn the inverse mapping from performance targets to mix designs. Rather than a "
            "single answer, we return a **spread of candidate mixes** consistent with the target.\n\n"
            "**How it works (see `docs/AMORTIZED_INFERENCE.md`):**\n"
            "1. **Amortized flow**: a BayesFlow normalizing flow is trained once, via simulation-based "
            "inference against the XGBoost forward model, to sample `p(mix | target strength)` "
            "instantly for any target.\n"
            "2. **Calibration**: the flow is checked with Simulation-Based Calibration (SBC); "
            "well-calibrated posteriors have uniform rank statistics.\n"
            "3. **GA fallback**: when the flow is untrained (or a carbon target is set), a transparent "
            "genetic-algorithm designer provides the same interface with no neural network.\n"
            "4. **Active experimental design**: `suggest_tests` blends target proximity with model "
            "uncertainty to point at high-value, under-explored 'empty spaces'.\n\n"
            "**Honest caveat:** the flow is trained against the forward *model*, not raw lab data, so "
            "its posterior reflects the model's view of the world (the 'simulation gap').\n"
        )

if __name__ == "__main__":
    explorer = BayesFlowExplorer()
    print(explorer.explain_empty_spaces())
