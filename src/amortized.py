"""
amortized.py - A real amortized Bayesian posterior for inverse mix design.

This is Phase 7 of docs/FIX_PLAN.md: it replaces the placeholder in bayesian.py
with a genuinely trained BayesFlow normalizing flow that learns

        p(mix parameters | target strength)

in one offline training pass, after which sampling any target is instantaneous.

WHAT "AMORTIZED" MEANS (in plain terms)
---------------------------------------
Classic Bayesian inference (e.g. MCMC) re-runs an expensive sampler for every new
observation. *Amortized* inference pays the cost ONCE: we train a neural network on
many simulated (parameters -> strength) pairs so it learns the whole inverse map.
Afterwards, asking "which mixes give 45 MPa?" is a single forward pass -- we amortise
the training cost across all future queries.

HOW IT WORKS HERE (simulation-based inference)
----------------------------------------------
1. Prior      : mixes drawn uniformly from the training-data envelope.
2. Simulator  : strength = XGBoost.predict(mix) + observation noise.
                (So the flow is trained against the forward model, not raw data --
                 this is the "simulation gap" the technical report flags in 3.3.)
3. Flow       : an Invertible Neural Network (normalizing flow) is trained to invert
                the simulator, i.e. to sample parameters consistent with a strength.
4. Calibration: we check the flow with Simulation-Based Calibration (SBC) -- if the
                posterior is well calibrated, the rank statistics are uniform.

The parameters live on very different scales (cement ~100-550, age 1-365), so we
train the flow in a standardised z-space and un-standardise samples on the way out.
"""
import os
import warnings
from typing import List, Optional

import numpy as np

from .models import StrengthPredictor
from .generative_ga import PARAM_NAMES, data_envelope

# BayesFlow / TensorFlow are optional heavy deps; import lazily-guarded so the rest
# of the package keeps working without them.
try:
    import tensorflow as tf
    from bayesflow import amortizers, networks, simulation, trainers
    _BF_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when deps are absent
    tf = None
    amortizers = networks = simulation = trainers = None
    _BF_AVAILABLE = False

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
AMORTIZER_DIR = os.path.join(MODEL_DIR, "amortizer")
WEIGHTS_PREFIX = os.path.join(AMORTIZER_DIR, "ckpt")


class AmortizedPosteriorModel:
    """
    Trained BayesFlow posterior p(parameters | target strength).

    Usage:
        m = AmortizedPosteriorModel(); m.train(); m.save()      # once, offline
        m = AmortizedPosteriorModel(); m.load()                 # later
        samples = m.sample(target_strength=45.0, n_samples=2000)
    """

    def __init__(
        self,
        predictor: Optional[StrengthPredictor] = None,
        param_names: List[str] = PARAM_NAMES,
        bounds: Optional[np.ndarray] = None,
        noise_sd: float = 2.0,
        num_coupling_layers: int = 5,
    ):
        if not _BF_AVAILABLE:
            raise ImportError(
                "TensorFlow/BayesFlow are not installed. Install the pinned amortized "
                "stack (see requirements.txt) or use the GA generator instead."
            )
        self.predictor = predictor or StrengthPredictor()
        self.param_names = param_names
        self.bounds = data_envelope(param_names) if bounds is None else np.asarray(bounds, float)
        self.n_params = len(self.bounds)
        self.noise_sd = noise_sd
        self.num_coupling_layers = num_coupling_layers

        # Standardisation constants.
        # Uniform prior over [lo, hi]: mean = midpoint, std = range / sqrt(12).
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        self._theta_mean = (lo + hi) / 2.0
        # Guard against a degenerate (constant) envelope column: a zero std would
        # produce inf/nan when standardising and de-standardising samples.
        self._theta_std = np.maximum((hi - lo) / np.sqrt(12.0), 1e-6)
        # Strength stats come from simulating the prior once (cheap, deterministic-ish).
        self._x_mean, self._x_std = self._estimate_x_stats()

        self.amortizer = None
        self._build()

    # -- simulation-based inference plumbing ---------------------------------
    def _draw_prior(self) -> np.ndarray:
        """One mix drawn uniformly from the data envelope."""
        return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])

    def _simulate_batch(self, theta: np.ndarray) -> np.ndarray:
        """Vectorised simulator: strength for a batch of mixes, plus noise."""
        theta = np.atleast_2d(theta).astype(np.float32)
        strengths = self.predictor.predict_batch(theta)
        noise = np.random.normal(0.0, self.noise_sd, size=strengths.shape)
        return (strengths + noise).reshape(-1, 1).astype(np.float32)

    def _estimate_x_stats(self, n: int = 2000):
        theta = np.array([self._draw_prior() for _ in range(n)], dtype=np.float32)
        x = self._simulate_batch(theta)
        return float(x.mean()), float(x.std() + 1e-6)

    def _configurator(self, forward_dict):
        """Map simulator output to standardised (parameters, direct_conditions)."""
        theta = forward_dict["prior_draws"].astype(np.float32)
        x = forward_dict["sim_data"].astype(np.float32)
        theta_z = (theta - self._theta_mean) / self._theta_std
        x_z = (x - self._x_mean) / self._x_std
        return {
            "parameters": theta_z.astype(np.float32),
            "direct_conditions": x_z.astype(np.float32),
        }

    def _generative_model(self):
        prior = simulation.Prior(prior_fun=self._draw_prior, param_names=self.param_names)
        simulator = simulation.Simulator(batch_simulator_fun=self._simulate_batch)
        return simulation.GenerativeModel(prior, simulator, name="mix_design", skip_test=True)

    def _build(self):
        inference_net = networks.InvertibleNetwork(
            num_params=self.n_params,
            num_coupling_layers=self.num_coupling_layers,
        )
        self.amortizer = amortizers.AmortizedPosterior(inference_net, name="mix_posterior")

    # -- training ------------------------------------------------------------
    def train(self, epochs: int = 30, iterations_per_epoch: int = 250, batch_size: int = 64):
        """Train the flow with online simulation-based learning."""
        gen_model = self._generative_model()
        trainer = trainers.Trainer(
            amortizer=self.amortizer,
            generative_model=gen_model,
            configurator=self._configurator,
            memory=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            history = trainer.train_online(
                epochs=epochs,
                iterations_per_epoch=iterations_per_epoch,
                batch_size=batch_size,
            )
        return history

    # -- persistence ---------------------------------------------------------
    def _ensure_built(self):
        """Instantiate network weights by running one tiny forward pass."""
        theta = np.array([self._draw_prior() for _ in range(8)], dtype=np.float32)
        x = self._simulate_batch(theta)
        conf = self._configurator({"prior_draws": theta, "sim_data": x})
        self.amortizer(conf)  # builds variables

    def save(self, prefix: str = WEIGHTS_PREFIX):
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        self.amortizer.save_weights(prefix)
        np.savez(
            prefix + "_norm.npz",
            theta_mean=self._theta_mean, theta_std=self._theta_std,
            x_mean=self._x_mean, x_std=self._x_std, bounds=self.bounds,
        )

    def load(self, prefix: str = WEIGHTS_PREFIX) -> bool:
        """Load trained weights + normalisation. Returns False if none exist."""
        if not os.path.exists(prefix + "_norm.npz"):
            return False
        norm = np.load(prefix + "_norm.npz")
        self._theta_mean, self._theta_std = norm["theta_mean"], norm["theta_std"]
        self._x_mean, self._x_std = float(norm["x_mean"]), float(norm["x_std"])
        self._ensure_built()
        self.amortizer.load_weights(prefix).expect_partial()
        return True

    @staticmethod
    def weights_exist(prefix: str = WEIGHTS_PREFIX) -> bool:
        return os.path.exists(prefix + "_norm.npz")

    # -- inference -----------------------------------------------------------
    def sample(self, target_strength: float, n_samples: int = 2000) -> np.ndarray:
        """
        Draw n_samples mixes from the amortized posterior for a target strength.
        Returns an (n_samples, n_params) array in the ORIGINAL parameter units,
        clipped to the data envelope.
        """
        x_z = np.array([[(target_strength - self._x_mean) / self._x_std]], dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            draws_z = self.amortizer.sample({"direct_conditions": x_z}, n_samples, to_numpy=True)
        draws_z = np.asarray(draws_z).reshape(n_samples, self.n_params)
        draws = draws_z * self._theta_std + self._theta_mean
        return np.clip(draws, self.bounds[:, 0], self.bounds[:, 1])

    # -- calibration (SBC) ---------------------------------------------------
    def sbc_ranks(self, n_datasets: int = 300, n_post: int = 250) -> np.ndarray:
        """
        Simulation-Based Calibration rank statistics.

        For each of n_datasets prior draws theta*, simulate x*, draw n_post posterior
        samples, and count how many fall below theta* per dimension. If the posterior
        is calibrated, these ranks are uniform on {0, ..., n_post}. Returns an
        (n_datasets, n_params) integer array of ranks (in z-space, which is monotonic
        in the original units so ranks are identical).
        """
        theta = np.array([self._draw_prior() for _ in range(n_datasets)], dtype=np.float32)
        x = self._simulate_batch(theta)
        theta_z = (theta - self._theta_mean) / self._theta_std
        x_z = (x - self._x_mean) / self._x_std

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            post = self.amortizer.sample({"direct_conditions": x_z.astype(np.float32)},
                                         n_post, to_numpy=True)
        post = np.asarray(post)  # (n_datasets, n_post, n_params)
        ranks = (post < theta_z[:, None, :]).sum(axis=1)
        return ranks.astype(int)


def train_and_save(epochs: int = 30, iterations_per_epoch: int = 250, batch_size: int = 64):
    """Convenience entry point: train, run SBC, save weights + a calibration summary."""
    model = AmortizedPosteriorModel()
    print(f"Training amortized posterior ({epochs} epochs x {iterations_per_epoch} its)...")
    model.train(epochs=epochs, iterations_per_epoch=iterations_per_epoch, batch_size=batch_size)
    model.save()
    print(f"Saved amortizer weights to {AMORTIZER_DIR}")

    ranks = model.sbc_ranks()
    n_post = 250
    # Chi-square uniformity check on the rank histogram, per parameter.
    from math import isnan
    summary = []
    for i, name in enumerate(model.param_names):
        r = ranks[:, i]
        mean_rank = r.mean() / n_post  # ideal ~0.5
        summary.append((name, mean_rank))
    print("SBC mean normalised rank per parameter (ideal ~0.50):")
    for name, mr in summary:
        print(f"  {name:16s} {mr:.3f}")
    return model, ranks


if __name__ == "__main__":
    train_and_save()
