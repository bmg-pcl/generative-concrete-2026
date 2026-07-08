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
from .data_fetcher import load_data


def dataset_fingerprint() -> int:
    """Row count of the current training dataset. Used to detect when a saved flow
    is stale relative to a re-calibrated dataset. -1 if the data can't be read."""
    try:
        return int(len(load_data()))
    except Exception:
        return -1


def model_fingerprint() -> str:
    """Content hash of the current forward strength model. The flow is trained
    against the forward model's outputs (SBI), so if the model changes — a new
    architecture (R7.1) or a Calibration retrain that leaves the row count the same
    — the flow no longer matches reality and must be treated as stale. Row count
    alone misses same-size retrains; this hash does not. Empty string if unreadable."""
    import hashlib
    from .models import QUANTILE_MODEL_PATH
    try:
        with open(QUANTILE_MODEL_PATH, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError:
        return ""

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

        # R7.2: the flow conditions on (strength, age) and outputs the OTHER 7 mix
        # parameters. Age is a design CONDITION, not a sampled latent — it is drawn
        # from the prior during training (so it is marginalised correctly) and
        # supplied at inference. This lets a fixed design age use the flow (before
        # R7.2 a pinned age routed to the GA); age=None recovers the old marginal by
        # drawing age from the prior per sample.
        self._age_idx = param_names.index("age")
        self._theta_cols = [i for i in range(self.n_params) if i != self._age_idx]
        self.n_theta = len(self._theta_cols)

        # Standardisation constants. Uniform prior over [lo, hi]: mean = midpoint,
        # std = range / sqrt(12). Guard degenerate (constant) columns.
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        mean_all = (lo + hi) / 2.0
        std_all = np.maximum((hi - lo) / np.sqrt(12.0), 1e-6)
        self._theta_mean = mean_all[self._theta_cols]     # 7-dim (non-age)
        self._theta_std = std_all[self._theta_cols]
        self._age_mean = float(mean_all[self._age_idx])   # scalar age condition
        self._age_std = float(std_all[self._age_idx])
        self._age_lo = float(lo[self._age_idx])
        self._age_hi = float(hi[self._age_idx])
        # Strength stats come from simulating the prior once (cheap, deterministic-ish).
        self._x_mean, self._x_std = self._estimate_x_stats()

        self.amortizer = None
        self._build()

    # -- age/theta split helpers --------------------------------------------
    def _split(self, draws: np.ndarray):
        """(N, 8) full draws -> (theta (N, 7), age (N, 1))."""
        draws = np.atleast_2d(draws)
        return draws[:, self._theta_cols], draws[:, self._age_idx:self._age_idx + 1]

    def _assemble(self, theta: np.ndarray, age: np.ndarray) -> np.ndarray:
        """(theta (N, 7), age (N,)) -> full (N, 8) mix vectors in canonical order."""
        full = np.empty((len(theta), self.n_params), dtype=float)
        full[:, self._theta_cols] = theta
        full[:, self._age_idx] = np.asarray(age, dtype=float).ravel()
        return full

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
        """Map simulator output to standardised (parameters, direct_conditions).

        parameters = the 7 non-age mix params; direct_conditions = [strength, age].
        The flow thus learns p(7 params | strength, age)."""
        draws = forward_dict["prior_draws"].astype(np.float32)
        x = forward_dict["sim_data"].astype(np.float32)
        theta, age = self._split(draws)
        theta_z = (theta - self._theta_mean) / self._theta_std
        age_z = (age - self._age_mean) / self._age_std
        x_z = (x - self._x_mean) / self._x_std
        cond = np.concatenate([x_z, age_z], axis=1)   # (N, 2)
        return {
            "parameters": theta_z.astype(np.float32),
            "direct_conditions": cond.astype(np.float32),
        }

    def _generative_model(self):
        prior = simulation.Prior(prior_fun=self._draw_prior, param_names=self.param_names)
        simulator = simulation.Simulator(batch_simulator_fun=self._simulate_batch)
        return simulation.GenerativeModel(prior, simulator, name="mix_design", skip_test=True)

    def _build(self):
        inference_net = networks.InvertibleNetwork(
            num_params=self.n_theta,   # 7 sampled params; age is a condition
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

    # Conditioning schema tag: bumped when the flow's inputs/outputs change so an
    # old-schema checkpoint is rejected (fall back to GA) instead of loaded wrongly.
    COND_SCHEMA = "strength_age_v2"

    def save(self, prefix: str = WEIGHTS_PREFIX):
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        self.amortizer.save_weights(prefix)
        np.savez(
            prefix + "_norm.npz",
            theta_mean=self._theta_mean, theta_std=self._theta_std,
            age_mean=self._age_mean, age_std=self._age_std,
            x_mean=self._x_mean, x_std=self._x_std, bounds=self.bounds,
            cond_schema=self.COND_SCHEMA,    # detect an incompatible conditioning schema (R7.2)
            n_train=dataset_fingerprint(),   # detect staleness after calibration
            model_hash=model_fingerprint(),  # detect a changed forward model (R7.1)
        )

    def load(self, prefix: str = WEIGHTS_PREFIX) -> bool:
        """
        Load trained weights + normalisation. Returns False if none exist OR if the
        flow is stale -- i.e. it was trained against a different dataset than the one
        now on disk (e.g. after Calibration appended lab rows and retrained XGBoost).
        The flow is conditioned on the *old* forward model, so a mismatch means it no
        longer matches reality; callers then fall back to the GA designer.
        """
        if not os.path.exists(prefix + "_norm.npz"):
            return False
        norm = np.load(prefix + "_norm.npz")
        # A checkpoint from an incompatible conditioning schema (e.g. the pre-R7.2
        # strength-only flow) has a different network shape; reject it cleanly.
        saved_schema = str(norm["cond_schema"]) if "cond_schema" in norm else ""
        if saved_schema != self.COND_SCHEMA:
            print("Amortized flow is STALE: conditioning schema changed (expected "
                  f"{self.COND_SCHEMA}). Retrain with `python -m src.amortized`; using the GA designer.")
            return False
        if "n_train" in norm:
            saved_n, current_n = int(norm["n_train"]), dataset_fingerprint()
            if saved_n != current_n:
                print(f"Amortized flow is STALE: trained on {saved_n} rows, dataset now has "
                      f"{current_n}. Retrain with `python -m src.amortized`; using the GA designer.")
                return False
        if "model_hash" in norm:
            saved_h, current_h = str(norm["model_hash"]), model_fingerprint()
            if current_h and saved_h != current_h:
                print("Amortized flow is STALE: the forward strength model changed since the "
                      "flow was trained. Retrain with `python -m src.amortized`; using the GA designer.")
                return False
        self._theta_mean, self._theta_std = norm["theta_mean"], norm["theta_std"]
        self._age_mean, self._age_std = float(norm["age_mean"]), float(norm["age_std"])
        self._x_mean, self._x_std = float(norm["x_mean"]), float(norm["x_std"])
        self._ensure_built()
        self.amortizer.load_weights(prefix).expect_partial()
        return True

    @staticmethod
    def weights_exist(prefix: str = WEIGHTS_PREFIX) -> bool:
        return os.path.exists(prefix + "_norm.npz")

    # -- inference -----------------------------------------------------------
    def sample(self, target_strength: float, n_samples: int = 2000,
               age: Optional[float] = None) -> np.ndarray:
        """
        Draw n_samples mixes from the amortized posterior for a target strength,
        conditioned on a design age. Returns an (n_samples, n_params) array in the
        ORIGINAL parameter units, clipped to the data envelope.

        age given  -> every sample is conditioned on that fixed age (R7.2: a pinned
                      design age now uses the flow instead of routing to the GA).
        age is None -> age is drawn from the prior per sample and conditioned on,
                      recovering the marginal p(mix | strength).
        """
        x_z = (float(target_strength) - self._x_mean) / self._x_std
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if age is None:
                # Marginalise over age: one age per sample, each its own condition.
                ages = np.random.uniform(self._age_lo, self._age_hi, n_samples)
                age_z = (ages - self._age_mean) / self._age_std
                cond = np.column_stack([np.full(n_samples, x_z), age_z]).astype(np.float32)
                draws_z = self.amortizer.sample({"direct_conditions": cond}, 1, to_numpy=True)
            else:
                ages = np.full(n_samples, float(age))
                age_z = (float(age) - self._age_mean) / self._age_std
                cond = np.array([[x_z, age_z]], dtype=np.float32)
                draws_z = self.amortizer.sample({"direct_conditions": cond}, n_samples, to_numpy=True)
        theta_z = np.asarray(draws_z).reshape(n_samples, self.n_theta)
        theta = theta_z * self._theta_std + self._theta_mean
        draws = self._assemble(theta, ages)
        return np.clip(draws, self.bounds[:, 0], self.bounds[:, 1])

    # -- calibration (SBC) ---------------------------------------------------
    def sbc_ranks(self, n_datasets: int = 300, n_post: int = 250) -> np.ndarray:
        """
        Simulation-Based Calibration rank statistics over the 7 sampled parameters.

        For each of n_datasets prior draws (theta*, age*), simulate x*, draw n_post
        posterior samples conditioned on (x*, age*), and count how many fall below
        theta* per sampled dimension. If the posterior is calibrated, these ranks are
        uniform on {0, ..., n_post}. Because (strength, age) conditions are drawn from
        the prior, the check spans the whole (strength × age) grid by construction.
        Returns an (n_datasets, n_theta) integer array of ranks (z-space ranks equal
        original-unit ranks, monotone per dimension).
        """
        draws = np.array([self._draw_prior() for _ in range(n_datasets)], dtype=np.float32)
        x = self._simulate_batch(draws)
        theta, age = self._split(draws)
        theta_z = (theta - self._theta_mean) / self._theta_std
        x_z = (x - self._x_mean) / self._x_std
        age_z = (age - self._age_mean) / self._age_std
        cond = np.concatenate([x_z, age_z], axis=1).astype(np.float32)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            post = self.amortizer.sample({"direct_conditions": cond}, n_post, to_numpy=True)
        post = np.asarray(post)  # (n_datasets, n_post, n_theta)
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
    # The flow samples the 7 non-age parameters (age is a condition, not sampled).
    theta_names = [model.param_names[i] for i in model._theta_cols]
    print("SBC mean normalised rank per sampled parameter (ideal ~0.50; age is a condition):")
    for i, name in enumerate(theta_names):
        print(f"  {name:16s} {ranks[:, i].mean() / n_post:.3f}")
    return model, ranks


if __name__ == "__main__":
    train_and_save()
