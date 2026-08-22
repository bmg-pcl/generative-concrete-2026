"""
src/properties.py -- the reusable "measured property" abstraction (R8.1).

R7.1 built ONE joint distributional model for strength (src/models.py:
StrengthPredictor) -- a multi-quantile XGBoost model, per-row rearrangement sorting
so the point estimate can never cross its own interval, split-conformal calibration
for measured (not assumed) coverage, and a kNN-distance novelty gate for honest
out-of-support detection. R8.1 is the SECOND measured property (workability / slump),
and the point of this module is to lift that pattern into something reusable, so every
future property (R8.3 carbon intervals, R9's joint posterior) follows the same shape
instead of re-deriving it.

UNITS AND ENVELOPE ARE PER-PROPERTY. A mix that is comfortably in-support for strength
(1030 rows, cement 102-540, SP 0-32.2 kg/m^3) can be far outside the slump corpus's
envelope (103 rows, cement 137-374, SP 4.4-19 kg/m^3) -- see
``tests/test_properties.py::test_strength_in_support_slump_out_of_support`` for an
exhibited example. Each ``PropertyModel`` instance carries its OWN support gate;
callers must ask the specific property whether it can speak for a given mix and must
never assume support carries over from another property's gate.

Convergence NOT done here (explicitly out of scope for R8.1, see docs/specs/R8.1
"Traps"): ``StrengthPredictor`` (src/models.py) is NOT refactored to inherit from
``PropertyModel`` in this package. That is a tempting but behaviour-risking change to
the repository's most load-bearing model. Converging the two -- once this second
instance has proven the abstraction is right -- is future work.
"""
import json
import os
from datetime import datetime, timezone
from typing import Optional, Sequence

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from .data_fetcher import load_slump_data
from .generative_ga import PARAM_NAMES
from .physical import workability_flag

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
SLUMP_MODEL_PATH = os.path.join(MODEL_DIR, "slump_model.json")

# Slump is measured on FRESH concrete -- there is no curing time to speak of, so
# (deliberately, see docs/specs/R8.1 point 3) the slump model's inputs are the first
# seven of PARAM_NAMES (the materials) and NOT the eighth ("age"). Do not add an age
# feature here; the slump corpus has no such column and inventing one would be a lie.
SLUMP_FEATURES = list(PARAM_NAMES[:7])
assert SLUMP_FEATURES == [
    "cement", "slag", "ash", "water", "superplasticizer", "coarse_agg", "fine_agg",
]

# 90% prediction-interval quantiles, same convention as src/models.py.
QUANTILES = np.array([0.05, 0.5, 0.95])


class PropertyModel:
    """One measured property: point estimate, conformal interval, and its OWN
    support gate.

    Architecturally identical to ``StrengthPredictor`` (src/models.py, R7.1): a single
    multi-quantile XGBoost model estimates (q05, q50, q95) jointly; per-row
    rearrangement sorting (Chernozhukov, Fernandez-Val & Galichon, 2010) makes the
    three monotone so the median point estimate can never fall outside its own
    interval; split-conformal calibration (Romano, Patterson & Candes, 2019) on a
    held-out fold gives the 90% interval MEASURED coverage rather than assumed
    coverage; a kNN-distance novelty gate, calibrated on a held-out fold's 95th
    percentile, flags mixes far from the training envelope.

    UNITS AND ENVELOPE ARE PER-PROPERTY. A mix in-support for one property may be far
    outside another's training envelope -- each instance carries its own gate, and
    callers must ask the property whether it can speak, never assume.

    Departures from ``StrengthPredictor``, forced by the small-data regime this class
    was built to handle (order 100 rows here, vs order 1000 for strength) -- see
    docs/specs/R8.1 "the three things that make this harder than it looks":

    - Far fewer, shallower trees (``n_estimators``, ``max_depth``): xgboost overfits a
      ~100-row corpus readily at the strength model's 800 trees / depth 6.
    - A smaller kNN neighbourhood (``support_k``): 103 rows split into an ~82-row
      training fold cannot support ``k=10`` the way 1030 rows can; the constructor
      also clamps ``k`` to ``len(fold) - 1`` so tiny folds never error.
    - No expectation that measured coverage lands near the nominal 90%: a calibration
      fold this small (order 20-30 rows) makes the coverage ESTIMATE itself noisy.
      Callers should report the measured number, not gate tightly around 0.90.
    """

    def __init__(
        self,
        name: str,
        feature_names: Sequence[str],
        n_estimators: int = 150,
        max_depth: int = 3,
        support_k: int = 5,
        test_size: float = 0.2,
        cal_size: float = 0.3,
        random_state: int = 42,
    ):
        self.name = name
        self.feature_names = list(feature_names)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.support_k = support_k
        self.test_size = test_size
        self.cal_size = cal_size
        self.random_state = random_state

        self._quantile: Optional[xgb.XGBRegressor] = None
        self._support: Optional[dict] = None
        # Held-out diagnostics from the most recent fit() (or a loaded save()).
        # NOT a gate inside this class -- callers/tests compare it against the
        # acceptance band. None until fit() or load() populates it.
        self.held_out_: Optional[dict] = None

    # -- training --------------------------------------------------------
    def fit(self, X, y) -> "PropertyModel":
        """Fit the joint quantile model, split-conformal calibrate it, and calibrate
        the kNN novelty gate -- all on disjoint folds of (X, y).

        Mirrors StrengthPredictor.train()'s fold structure (fit / calibration / test,
        held-out coverage measured on the test fold) but is data-agnostic: this method
        knows nothing about slump or strength, only that it is given a feature matrix
        and a target.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(X)
        if n < 20:
            raise ValueError(
                f"PropertyModel({self.name!r}): {n} rows is too few to fit and "
                "conformal-calibrate a held-out split; refusing to produce a model "
                "that cannot report honest coverage."
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        # Split-conformal validity requires the deployed model to be fit on a fold
        # disjoint from the one that calibrates its interval (Romano et al. 2019).
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X_train, y_train, test_size=self.cal_size, random_state=self.random_state
        )

        quantile = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=QUANTILES,
            n_estimators=self.n_estimators,
            learning_rate=0.05,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        quantile.fit(X_fit, y_fit)
        self._quantile = quantile

        # CQR conformal correction: symmetric conformity score on the calibration
        # fold, using the SAME monotone (sorted) quantiles the interval will report.
        qp = np.sort(np.atleast_2d(quantile.predict(X_cal)), axis=1)
        q_lo, q_hi = qp[:, 0], qp[:, 2]
        scores = np.maximum(q_lo - y_cal, y_cal - q_hi)
        n_cal = len(scores)
        k_conf = int(np.ceil((n_cal + 1) * 0.90))
        conformal_q = float(np.sort(scores)[min(k_conf, n_cal) - 1])

        # Out-of-support reference: standardise the TRAINING fold's features (same
        # convention as StrengthPredictor -- the full 80% training split, not just
        # the smaller fit sub-fold) and record the typical distance to the k-th
        # nearest training neighbour.
        mean, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-9
        Xs = (X_train - mean) / std
        k_eff = min(self.support_k, len(Xs) - 1)
        nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(Xs)
        ref = float(np.median(nn.kneighbors(Xs)[0][:, k_eff]))  # col 0 is self (dist 0)

        self._support = {
            "Xs": Xs, "mean": mean, "std": std, "ref": ref, "k": k_eff,
            "conformal_q": conformal_q, "novelty_threshold": 1.5,
        }
        # Novelty threshold: 95th percentile of the held-out TEST fold's novelty
        # against the training support -- genuine unseen in-distribution mixes
        # calibrate the cutoff, same as StrengthPredictor.
        novelty_threshold = float(np.percentile(self.novelty(X_test), 95))
        self._support["novelty_threshold"] = novelty_threshold

        # Measured (not assumed) coverage and interval width on the held-out test
        # fold -- report these, do not gate tightly, do not tune toward a target.
        lo, med, hi = self.predict_interval(X_test)
        coverage = float(np.mean((y_test >= lo) & (y_test <= hi)))
        median_width = float(np.median(hi - lo))
        self.held_out_ = {
            "n_test": int(len(X_test)),
            "n_cal": int(n_cal),
            "coverage": coverage,
            "median_interval_width": median_width,
        }
        return self

    # -- lazy accessors ----------------------------------------------------
    def _predict_quantiles(self, X) -> np.ndarray:
        """(N, 3) monotone quantile predictions (lo, med, hi) via rearrangement."""
        if self._quantile is None:
            raise ValueError(f"PropertyModel({self.name!r}) not fitted/loaded.")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        p = np.atleast_2d(self._quantile.predict(X))
        return np.sort(p, axis=1)

    def support_threshold(self) -> float:
        """The data-calibrated novelty cutoff (fallback 1.5 if never fit)."""
        return self._support["novelty_threshold"] if self._support else 1.5

    # -- point prediction ----------------------------------------------------
    def predict(self, X):
        """Point estimate (the model's median). Scalar for a single (1D) mix,
        array for a batch (2D)."""
        X_arr = np.asarray(X, dtype=float)
        med = self._predict_quantiles(X_arr)[:, 1]
        return float(med[0]) if X_arr.ndim == 1 else med

    # -- uncertainty ---------------------------------------------------------
    def predict_interval(self, X):
        """Return (lo, med, hi) arrays for a conformalised 90% prediction interval
        (always arrays, even for a single 1D input -- same convention as
        StrengthPredictor.predict_interval).

        Because the point estimate IS the median of the same sorted quantiles,
        ``lo - q_hat <= med <= hi + q_hat`` always holds (q_hat >= 0): the point
        estimate can never cross its own interval.
        """
        q = self._predict_quantiles(X)
        c = self._support["conformal_q"] if self._support else 0.0
        return q[:, 0] - c, q[:, 1], q[:, 2] + c

    # -- out-of-support (extrapolation) detection ----------------------------
    def novelty(self, X) -> np.ndarray:
        """Novelty score per row: distance to the k-th nearest training point in
        standardised space, divided by the typical in-training distance.

        ~1.0 means as close to the data as a typical training point; values well
        above 1 mean the row sits in a sparsely-sampled / extrapolated region for
        THIS property specifically. Returns zeros if the model has no support data.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        sup = self._support
        if sup is None:
            return np.zeros(len(X))
        xs = (X - sup["mean"]) / sup["std"]
        Xs = sup["Xs"]
        out = np.empty(len(xs))
        for i in range(0, len(xs), 256):
            d = np.linalg.norm(xs[i:i + 256, None, :] - Xs[None, :, :], axis=2)
            out[i:i + 256] = np.partition(d, sup["k"], axis=1)[:, sup["k"]]
        return out / sup["ref"]

    def in_support(self, X, threshold: float = None) -> np.ndarray:
        """Boolean mask: True where a row is within THIS property's trusted region.

        Uses the data-calibrated threshold by default (see support_threshold()).
        """
        if threshold is None:
            threshold = self.support_threshold()
        return self.novelty(X) <= threshold

    # -- persistence -----------------------------------------------------
    def save(self, path: str):
        """Serialise the whole model -- quantile booster, support/novelty data,
        hyperparameters, and held-out diagnostics -- into ONE committed JSON file.

        The xgboost booster is embedded as its native JSON representation (a nested
        object), not a side-car file, so ``models/slump_model.json`` is the single
        artifact the spec asks for.
        """
        if self._quantile is None or self._support is None:
            raise ValueError(f"PropertyModel({self.name!r}): nothing to save, fit() first.")
        booster_json = json.loads(
            self._quantile.get_booster().save_raw(raw_format="json").decode("utf-8")
        )
        sup = self._support
        payload = {
            "name": self.name,
            "feature_names": self.feature_names,
            "hyperparams": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "support_k": self.support_k,
                "test_size": self.test_size,
                "cal_size": self.cal_size,
                "random_state": self.random_state,
            },
            "booster": booster_json,
            "support": {
                "Xs": sup["Xs"].tolist(),
                "mean": sup["mean"].tolist(),
                "std": sup["std"].tolist(),
                "ref": sup["ref"],
                "k": sup["k"],
                "conformal_q": sup["conformal_q"],
                "novelty_threshold": sup["novelty_threshold"],
            },
            "held_out": self.held_out_,
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        out_dir = os.path.dirname(path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PropertyModel":
        with open(path) as f:
            payload = json.load(f)
        obj = cls(
            name=payload["name"],
            feature_names=payload["feature_names"],
            **payload["hyperparams"],
        )
        booster_bytes = json.dumps(payload["booster"]).encode("utf-8")
        quantile = xgb.XGBRegressor()
        quantile.load_model(bytearray(booster_bytes))
        obj._quantile = quantile
        sup = payload["support"]
        obj._support = {
            "Xs": np.array(sup["Xs"], dtype=float),
            "mean": np.array(sup["mean"], dtype=float),
            "std": np.array(sup["std"], dtype=float),
            "ref": float(sup["ref"]),
            "k": int(sup["k"]),
            "conformal_q": float(sup["conformal_q"]),
            "novelty_threshold": float(sup["novelty_threshold"]),
        }
        obj.held_out_ = payload.get("held_out")
        return obj


# ---------------------------------------------------------------------------
# Slump-specific wiring: the first consumer of PropertyModel.
# ---------------------------------------------------------------------------

# Fewer/shallower trees and a smaller k than StrengthPredictor's 800/depth-6/k=10 --
# a deliberate departure for a ~100-row corpus, not an oversight. See the
# PropertyModel docstring and docs/specs/R8.1 for why. random_state=42 matches the
# convention used everywhere else in this repo (src/models.py, src/generative_ga.py).
SLUMP_HYPERPARAMS = dict(
    n_estimators=150, max_depth=3, support_k=5,
    test_size=0.2, cal_size=0.3, random_state=42,
)


def slump_data_envelope() -> np.ndarray:
    """Per-feature (min, max) bounds from the SLUMP corpus's own 7 material columns.

    Deliberately NOT ``generative_ga.data_envelope()`` -- that reads the 1030-row
    STRENGTH corpus and includes an ``age`` column the slump model does not use.
    Reusing it would silently mix two different corpora's envelopes. See the
    PropertyModel docstring's "UNITS AND ENVELOPE ARE PER-PROPERTY" note.
    """
    df = load_slump_data()
    return np.array([(df[c].min(), df[c].max()) for c in SLUMP_FEATURES], dtype=float)


def train_slump_model(save: bool = True) -> PropertyModel:
    """Train the slump PropertyModel on the committed corpus's 7 material columns ->
    slump_cm, and (by default) save it to models/slump_model.json.

    Does NOT train on, or blend in, the corpus's own ``Compressive Strength`` column
    -- ``load_slump_data()`` loads it for completeness but this function never reads
    it (see that function's docstring for why: a different 103-row experiment from a
    different lab than the 1030-row strength corpus).
    """
    df = load_slump_data()
    X = df[SLUMP_FEATURES].to_numpy(dtype=float)
    y = df["slump_cm"].to_numpy(dtype=float)
    model = PropertyModel(name="slump_cm", feature_names=SLUMP_FEATURES, **SLUMP_HYPERPARAMS)
    model.fit(X, y)
    if save:
        model.save(SLUMP_MODEL_PATH)
    return model


_slump_model_singleton: Optional[PropertyModel] = None


def get_slump_model() -> PropertyModel:
    """Lazily load the committed slump model. Never trains or downloads at import
    or call time -- if the artifact is missing, run ``python -m src.properties``."""
    global _slump_model_singleton
    if _slump_model_singleton is None:
        if not os.path.exists(SLUMP_MODEL_PATH):
            raise FileNotFoundError(
                f"{SLUMP_MODEL_PATH} not found. Run `python -m src.properties` to "
                "train and save it (this repo commits the artifact; it should already "
                "exist on a normal checkout)."
            )
        _slump_model_singleton = PropertyModel.load(SLUMP_MODEL_PATH)
    return _slump_model_singleton


def slump_estimate(mix: dict) -> dict:
    """Honest workability estimate for a mix design -- the integrator-facing entry
    point named in docs/specs/R8.1.

    Tries the measured SlumpModel first. If the mix falls outside ITS OWN support
    envelope (which is not the same envelope the strength model uses -- see
    PropertyModel's docstring), this degrades to ``physical.workability_flag``'s
    labelled heuristic rather than returning a confident model number outside the
    training envelope. ``physical.workability_flag`` is frozen for this package: it
    is called here, never edited.

    Returns a dict with keys ``slump_cm``, ``lo``, ``hi``, ``in_support``, ``basis``
    (``"model"`` or ``"heuristic"``), and ``reason`` (populated only when heuristic).
    On the heuristic path ``slump_cm``/``lo``/``hi`` are ``None`` -- never a bare
    number produced by a model that was not asked to speak for this region.
    """
    model = get_slump_model()
    x = np.array([float(mix.get(f, 0.0)) for f in SLUMP_FEATURES], dtype=float)
    in_support = bool(model.in_support(x)[0])

    if in_support:
        lo, med, hi = model.predict_interval(x)
        return {
            "slump_cm": float(med[0]), "lo": float(lo[0]), "hi": float(hi[0]),
            "in_support": True, "basis": "model", "reason": None,
        }

    heuristic_flag = workability_flag(mix)
    reason = heuristic_flag or (
        "Mix lies outside the slump corpus's training envelope (103 rows); no "
        "measured estimate is available for this region."
    )
    return {
        "slump_cm": None, "lo": None, "hi": None,
        "in_support": False, "basis": "heuristic", "reason": reason,
    }


if __name__ == "__main__":
    m = train_slump_model()
    print(f"Trained slump PropertyModel. Held-out diagnostics: {m.held_out_}")
    print(f"Saved to {SLUMP_MODEL_PATH}")
