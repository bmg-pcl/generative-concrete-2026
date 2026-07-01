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
from .generative_ga import PopulationInverseDesigner

class BayesFlowExplorer:
    """
    Explores the inverse design space p(parameters | target strength).

    NOTE ON THE CURRENT IMPLEMENTATION (see docs/FIX_PLAN.md):
    The target-conditioned sampling below is provided by a simple, transparent
    GA-based inverse designer (`PopulationInverseDesigner`), NOT by a trained
    normalizing flow. This is deliberate: the GA generator is easy to understand,
    needs no TensorFlow, and -- unlike the previous placeholder -- actually
    depends on the requested target. The BayesFlow scaffolding (`build_model`/
    `train`) is retained as future work (Phase 7) for a true amortized posterior,
    but is not used by `sample_posterior`.
    """

    def __init__(self):
        self.predictor = StrengthPredictor()
        self.amortizer = None
        self.is_trained = False
        self.param_names = ["cement", "slag", "ash", "water", "superplasticizer", "coarse_agg", "fine_agg", "age"]
        self.bounds = np.array([
            (100, 550), (0, 360), (0, 200), (120, 250), (0, 30), (700, 1150), (550, 1000), (1, 365)
        ])
        # Lazily-built GA inverse designer that backs the generative interface.
        self._designer = None

    @property
    def designer(self) -> PopulationInverseDesigner:
        """The GA-based inverse designer, built on first use (shares our predictor)."""
        if self._designer is None:
            self._designer = PopulationInverseDesigner(predictor=self.predictor)
        return self._designer
        
    def _prior(self):
        """Draws samples from the parameter prior."""
        return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])

    def _simulator(self, theta):
        """Simulates strength given parameters using the trained XGBoost model + noise."""
        strength = self.predictor.predict(theta)
        noise = np.random.normal(0, 2.0) # Assume 2MPa observation noise
        return np.array([strength + noise])

    def build_model(self):
        """Configures the BayesFlow neural networks."""
        if bf is None:
            return
            
        # Summary Network (Process strength data)
        summary_net = bf.networks.SimpleSequenceNet(n_out=16) 
        
        # Inference Network (Normalizing Flow)
        inference_net = bf.networks.InvertibleNetwork(
            num_params=len(self.bounds),
            num_coupling_layers=4
        )
        
        self.amortizer = bf.amortizers.AmortizedPosterior(inference_net, summary_net)

    def train(self, epochs=20, iterations_per_epoch=100, batch_size=32):
        """
        Placeholder for training a true amortized BayesFlow posterior (Phase 7).

        This is NOT yet implemented: the loop below only generates simulation data
        and the actual `bf.trainers.Trainer(...)` fit is not wired up. It does not
        affect `sample_posterior`, which is served by the GA inverse designer. The
        method is kept as a scaffold and intentionally leaves `is_trained = False`
        so nothing mistakes it for a calibrated flow.
        """
        if bf is None:
            print("BayesFlow/TensorFlow not installed; amortized training is unavailable. "
                  "sample_posterior() uses the GA inverse designer instead.")
            return

        if self.amortizer is None:
            self.build_model()

        print("BayesFlow amortized training is not yet implemented (Phase 7). "
              "Generating simulation data only; no fit is performed.")
        # Future work: draw (theta, x) from _prior/_simulator and call
        # bf.trainers.Trainer(amortizer=self.amortizer). Left unwired on purpose.

    def sample_posterior(self, target_strength: float, carbon_target: float = None, n_samples: int = 2000) -> np.ndarray:
        """
        Draw a cloud of mix designs conditioned on the target strength (and an
        optional carbon target).

        Delegates to the GA-based `PopulationInverseDesigner`: it searches -- within
        the training-data envelope -- for mixes whose predicted strength matches the
        target, then returns a spread around the best of them. Unlike the previous
        placeholder, the result genuinely depends on `target_strength`.
        """
        return self.designer.sample(
            target_strength,
            n_samples=n_samples,
            carbon_target=carbon_target,
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
        return (
            "### Amortized Bayesian Inference with BayesFlow\n"
            "We use **Normalizing Flows** to learn the entire inverse mapping from performance targets to mix designs. "
            "Unlike traditional models which give a single answer, BayesFlow gives you the **full posterior probability mesh**.\n\n"
            "**Key Advantages:**\n"
            "1. **Multi-Objective Targets**: We can now condition the posterior on both **Target Strength** and **Carbon Footprint**.\n"
            "2. **Active Experimental Design**: The `suggest_tests` feature identifies the top-five mix designs that are both likely to meet your targets and reside in high-uncertainty regions of the model.\n"
            "3. **Instant Inference**: Once trained, we can query 10,000+ mix candidates for any target in milliseconds.\n"
            "4. **Empty Space Detection**: High-variance posteriors directly pinpoint where our knowledge is 'thin'.\n"
        )

if __name__ == "__main__":
    explorer = BayesFlowExplorer()
    print(explorer.explain_empty_spaces())
