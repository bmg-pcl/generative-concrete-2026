import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
import numpy as np
from .data_fetcher import load_data
import os
import json
from datetime import datetime, timezone

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
QUANTILE_MODEL_PATH = os.path.join(MODEL_DIR, "strength_quantiles.json")
SUPPORT_PATH = os.path.join(MODEL_DIR, "support.npz")
METRICS_HISTORY_PATH = os.path.join(MODEL_DIR, "metrics_history.json")


def _append_metrics_history(record: dict):
    """Append a training record to the (runtime, gitignored) metrics history so the
    Calibration tab can show before/after accuracy as lab data is added."""
    try:
        history = []
        if os.path.exists(METRICS_HISTORY_PATH):
            with open(METRICS_HISTORY_PATH) as f:
                history = json.load(f)
        history.append(record)
        with open(METRICS_HISTORY_PATH, "w") as f:
            json.dump(history, f, indent=2)
    except (OSError, ValueError):
        pass  # history is a nice-to-have, never fatal


def load_metrics_history() -> list:
    try:
        with open(METRICS_HISTORY_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return []

# 90% prediction-interval quantiles and the k for out-of-support kNN distance.
QUANTILES = np.array([0.05, 0.5, 0.95])
SUPPORT_K = 10


class StrengthPredictor:
    """One joint distributional strength model (roadmap R7.1).

    A single XGBoost multi-quantile model estimates (q05, q50, q95). The MEDIAN is
    the point prediction and the outer quantiles (conformalised — see
    ``predict_interval``) are the 90% interval, so the point estimate and the
    interval come from ONE distribution. Per-row rearrangement sorting
    (Chernozhukov, Fernández-Val & Galichon, 2010) makes the three quantiles
    monotone, so the point prediction can never fall outside its own interval —
    the mean/quantile crossing pathology of the previous two-model architecture
    (docs/PAPER.md §8.3) is impossible by construction.
    """

    def __init__(self):
        self._quantile = None      # the joint multi-quantile model (lazy)
        self._support = None       # kNN support data (lazy)
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

    # -- training ------------------------------------------------------------
    def train(self):
        """Train the joint quantile model (point prediction + intervals) and the
        out-of-support kNN reference on the current dataset.

        Split-conformal validity requires the deployed model to be fit on a proper
        training fold and calibrated on a disjoint fold, so the model is fit on 75%
        of the 80% training split and the remaining 25% calibrates the interval."""
        df = load_data()
        X = df.drop("strength", axis=1)
        y = df["strength"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Joint model on a FIT fold, conformalised on a held-out CALIBRATION fold
        # (split conformal / CQR — Romano, Patterson & Candès 2019) so the 90% interval
        # has *measured* coverage rather than relying on the raw trained quantiles.
        X_fit, X_cal, y_fit, y_cal = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
        # 800 trees (vs the old mean model's 500): the pinball objective fits three
        # quantiles jointly and on the 75% fit fold, so it wants a little more
        # capacity to match the old point accuracy.
        quantile = xgb.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=QUANTILES,
            n_estimators=800, learning_rate=0.05, max_depth=6, random_state=42,
        )
        quantile.fit(X_fit, y_fit)
        quantile.save_model(QUANTILE_MODEL_PATH)
        self._quantile = quantile

        # Point-prediction accuracy: the MEDIAN of the joint model on the test fold.
        y_pred = self.predict_batch(np.asarray(X_test, dtype=float))
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"Model Trained (joint quantile). RMSE: {rmse:.2f}, R2: {r2:.2f}")

        # CQR correction: symmetric conformity score on the calibration fold, using
        # the SAME monotone (sorted) quantiles the interval will report.
        qp = np.sort(np.atleast_2d(quantile.predict(X_cal)), axis=1)
        q_lo, q_hi = qp[:, 0], qp[:, 2]
        yc = np.asarray(y_cal, dtype=float)
        scores = np.maximum(q_lo - yc, yc - q_hi)
        n_cal = len(scores)
        k = int(np.ceil((n_cal + 1) * 0.90))
        conformal_q = float(np.sort(scores)[min(k, n_cal) - 1])

        # Out-of-support reference: standardise the training features and record the
        # typical distance to the k-th nearest training neighbour.
        Xtr = np.asarray(X_train, dtype=float)
        mean, std = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-9
        Xs = (Xtr - mean) / std
        nn = NearestNeighbors(n_neighbors=SUPPORT_K + 1).fit(Xs)
        ref = float(np.median(nn.kneighbors(Xs)[0][:, SUPPORT_K]))  # kth dist (col 0 is self)
        self._support = {"Xs": Xs, "mean": mean, "std": std, "ref": ref, "k": SUPPORT_K,
                         "conformal_q": conformal_q, "novelty_threshold": 1.5}

        # Novelty threshold: 95th percentile of the held-out test fold's novelty against
        # the training support — genuine unseen in-distribution mixes calibrate the cutoff.
        novelty_threshold = float(np.percentile(self.novelty(np.asarray(X_test, dtype=float)), 95))
        self._support["novelty_threshold"] = novelty_threshold

        np.savez(SUPPORT_PATH, Xs=Xs, mean=mean, std=std, ref=ref, k=SUPPORT_K,
                 conformal_q=conformal_q, novelty_threshold=novelty_threshold)

        # Measured coverage of the (now conformalised) 90% interval on the test set.
        lo, _, hi = self.predict_interval(np.asarray(X_test, dtype=float))
        yt = np.asarray(y_test, dtype=float)
        coverage = float(np.mean((yt >= lo) & (yt <= hi)))
        print(f"Conformal q={conformal_q:.2f} | 90% PI coverage(test)={coverage:.3f} | "
              f"novelty threshold={novelty_threshold:.2f}")

        _append_metrics_history({
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_rows": int(len(X)), "rmse": float(rmse), "r2": float(r2),
        })
        return {"rmse": rmse, "r2": r2, "coverage": coverage,
                "conformal_q": conformal_q, "novelty_threshold": novelty_threshold}

    # -- lazy loaders --------------------------------------------------------
    def _load_quantile(self):
        if self._quantile is None:
            if os.path.exists(QUANTILE_MODEL_PATH):
                self._quantile = xgb.XGBRegressor()
                self._quantile.load_model(QUANTILE_MODEL_PATH)
            else:
                raise ValueError("Model not trained. Run train() first.")
        return self._quantile

    def _predict_quantiles(self, X: np.ndarray) -> np.ndarray:
        """(N, 3) monotone quantile predictions (lo, med, hi) via rearrangement."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        p = np.atleast_2d(self._load_quantile().predict(X))
        return np.sort(p, axis=1)

    def _load_support(self):
        if self._support is None and os.path.exists(SUPPORT_PATH):
            d = np.load(SUPPORT_PATH)
            self._support = {"Xs": d["Xs"], "mean": d["mean"], "std": d["std"],
                             "ref": float(d["ref"]), "k": int(d["k"]),
                             "conformal_q": float(d["conformal_q"]) if "conformal_q" in d else 0.0,
                             "novelty_threshold": float(d["novelty_threshold"]) if "novelty_threshold" in d else 1.5}
        return self._support

    def support_threshold(self) -> float:
        """The data-calibrated novelty cutoff (fallback 1.5 for old artifacts)."""
        sup = self._load_support()
        return sup["novelty_threshold"] if sup else 1.5

    # -- point prediction ----------------------------------------------------
    def predict(self, mix_design: np.ndarray) -> float:
        """Predict strength for a single mix design (the model's median)."""
        mix_design = np.asarray(mix_design, dtype=float)
        if mix_design.ndim == 1:
            mix_design = mix_design.reshape(1, -1)
        return float(self._predict_quantiles(mix_design)[0, 1])

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict strength for a batch of mixes; returns a 1D array (len N)."""
        return self._predict_quantiles(X)[:, 1]

    # -- uncertainty ---------------------------------------------------------
    def predict_interval(self, X: np.ndarray):
        """Return (lo, med, hi) arrays for a conformalised 90% prediction interval.

        Applies the split-conformal correction q̂ (from calibration) to the monotone
        quantile bounds so coverage is calibrated. Because the point prediction IS
        the median of the same sorted quantiles, ``lo − q̂ ≤ med ≤ hi + q̂`` always
        holds (q̂ ≥ 0) — the point prediction can never cross its own interval.
        """
        q = self._predict_quantiles(X)
        sup = self._load_support()
        c = sup["conformal_q"] if sup else 0.0
        return q[:, 0] - c, q[:, 1], q[:, 2] + c

    def predict_variance(self, mix_design: np.ndarray) -> float:
        """Uncertainty proxy: half the 90% prediction-interval width (MPa)."""
        lo, _, hi = self.predict_interval(np.atleast_2d(mix_design))
        return float((hi[0] - lo[0]) / 2.0)

    # -- out-of-support (extrapolation) detection ----------------------------
    def novelty(self, X: np.ndarray) -> np.ndarray:
        """Novelty score per mix: distance to the k-th nearest training point in
        standardised space, divided by the typical in-training distance.

        ~1.0 means as close to the data as a typical training point; values well
        above 1 mean the mix sits in a sparsely-sampled / extrapolated region, so the
        prediction is less trustworthy however confident. Returns zeros if no support
        data is available.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        sup = self._load_support()
        if sup is None:
            return np.zeros(len(X))
        xs = (X - sup["mean"]) / sup["std"]
        # Pairwise distances to the training set (chunked to bound memory).
        Xs = sup["Xs"]
        out = np.empty(len(xs))
        for i in range(0, len(xs), 256):
            d = np.linalg.norm(xs[i:i + 256, None, :] - Xs[None, :, :], axis=2)
            out[i:i + 256] = np.partition(d, sup["k"], axis=1)[:, sup["k"]]
        return out / sup["ref"]

    def in_support(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        """Boolean mask: True where a mix is within the trusted data region.

        Uses the data-calibrated threshold by default (see support_threshold()).
        """
        if threshold is None:
            threshold = self.support_threshold()
        return self.novelty(X) <= threshold


if __name__ == "__main__":
    predictor = StrengthPredictor()
    predictor.train()
