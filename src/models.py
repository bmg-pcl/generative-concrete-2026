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
STRENGTH_MODEL_PATH = os.path.join(MODEL_DIR, "strength_model.json")
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
    def __init__(self):
        self.model = None
        self._quantile = None      # XGBoost multi-quantile regressor (lazy)
        self._support = None       # kNN support data (lazy)
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

    # -- training ------------------------------------------------------------
    def train(self):
        """Train the mean model, a quantile model (prediction intervals), and the
        out-of-support kNN reference, all on the current dataset."""
        df = load_data()
        X = df.drop("strength", axis=1)
        y = df["strength"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Mean model: trained on the FULL 80% training split (unchanged).
        self.model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"Model Trained. RMSE: {rmse:.2f}, R2: {r2:.2f}")
        self.model.save_model(STRENGTH_MODEL_PATH)

        # Quantile model on a FIT fold, conformalised on a held-out CALIBRATION fold
        # (split conformal / CQR — Romano, Patterson & Candès 2019) so the 90% interval
        # has *measured* coverage rather than relying on the raw trained quantiles.
        X_fit, X_cal, y_fit, y_cal = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
        quantile = xgb.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=QUANTILES,
            n_estimators=400, learning_rate=0.05, max_depth=4, random_state=42,
        )
        quantile.fit(X_fit, y_fit)
        quantile.save_model(QUANTILE_MODEL_PATH)
        self._quantile = quantile

        # CQR correction: symmetric conformity score on the calibration fold.
        qp = np.atleast_2d(quantile.predict(X_cal))
        q_lo, q_hi = np.minimum(qp[:, 0], qp[:, 2]), np.maximum(qp[:, 0], qp[:, 2])
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
    def _ensure_model(self):
        if self.model is None:
            if os.path.exists(STRENGTH_MODEL_PATH):
                self.model = xgb.XGBRegressor()
                self.model.load_model(STRENGTH_MODEL_PATH)
            else:
                raise ValueError("Model not trained. Run train() first.")
        return self.model

    def _load_quantile(self):
        if self._quantile is None and os.path.exists(QUANTILE_MODEL_PATH):
            self._quantile = xgb.XGBRegressor()
            self._quantile.load_model(QUANTILE_MODEL_PATH)
        return self._quantile

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
        """Predict strength for a single mix design."""
        model = self._ensure_model()
        if mix_design.ndim == 1:
            mix_design = mix_design.reshape(1, -1)
        return float(model.predict(mix_design)[0])

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict strength for a batch of mixes; returns a 1D array (len N)."""
        model = self._ensure_model()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.asarray(model.predict(X), dtype=float).ravel()

    # -- uncertainty ---------------------------------------------------------
    def predict_interval(self, X: np.ndarray):
        """Return (lo, med, hi) arrays for a conformalised 90% prediction interval.

        Applies the split-conformal correction q̂ (from calibration) to the quantile
        bounds so coverage is calibrated. Falls back to a flat ±4 MPa band when no
        quantile model is available.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        qm = self._load_quantile()
        if qm is None:
            med = self.predict_batch(X)
            return med - 4.0, med, med + 4.0
        p = np.atleast_2d(qm.predict(X))
        lo = np.minimum(p[:, 0], p[:, 2])
        hi = np.maximum(p[:, 0], p[:, 2])
        sup = self._load_support()
        q = sup["conformal_q"] if sup else 0.0
        return lo - q, p[:, 1], hi + q

    def predict_variance(self, mix_design: np.ndarray) -> float:
        """Uncertainty proxy: half the 90% prediction-interval width (MPa).

        Falls back to a heuristic when no quantile model is available.
        """
        if self._load_quantile() is None:
            return self._heuristic_variance(mix_design)
        lo, _, hi = self.predict_interval(np.atleast_2d(mix_design))
        return float((hi[0] - lo[0]) / 2.0)

    def _heuristic_variance(self, mix_design: np.ndarray) -> float:
        cement = mix_design[0] if mix_design.ndim == 1 else mix_design[0, 0]
        water = mix_design[3] if mix_design.ndim == 1 else mix_design[0, 3]
        slag = mix_design[1] if mix_design.ndim == 1 else mix_design[0, 1]
        ash = mix_design[2] if mix_design.ndim == 1 else mix_design[0, 2]
        w_c = water / max(cement, 1)
        scm_ratio = (slag + ash) / max(cement, 1)
        variance = 2.0
        if w_c < 0.3 or w_c > 0.6:
            variance += 2.0
        if scm_ratio > 0.4:
            variance += 1.5
        return variance

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
