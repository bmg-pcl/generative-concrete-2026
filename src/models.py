import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
from .data_fetcher import load_data
import os

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
STRENGTH_MODEL_PATH = os.path.join(MODEL_DIR, "strength_model.json")
QUANTILE_MODEL_PATH = os.path.join(MODEL_DIR, "strength_quantiles.json")
SUPPORT_PATH = os.path.join(MODEL_DIR, "support.npz")

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

        self.model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"Model Trained. RMSE: {rmse:.2f}, R2: {r2:.2f}")
        self.model.save_model(STRENGTH_MODEL_PATH)

        # Quantile model → genuine per-mix prediction intervals (heteroscedastic:
        # wider where the data is noisy/sparse) instead of a hand-tuned heuristic.
        quantile = xgb.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=QUANTILES,
            n_estimators=400, learning_rate=0.05, max_depth=4, random_state=42,
        )
        quantile.fit(X_train, y_train)
        quantile.save_model(QUANTILE_MODEL_PATH)
        self._quantile = quantile

        # Out-of-support reference: standardise the training features and record the
        # typical distance to the k-th nearest training neighbour. A query far from
        # the data (high kNN distance) is extrapolation, however confident the model.
        Xtr = np.asarray(X_train, dtype=float)
        mean, std = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-9
        Xs = (Xtr - mean) / std
        nn = NearestNeighbors(n_neighbors=SUPPORT_K + 1).fit(Xs)
        ref = float(np.median(nn.kneighbors(Xs)[0][:, SUPPORT_K]))  # kth dist (col 0 is self)
        np.savez(SUPPORT_PATH, Xs=Xs, mean=mean, std=std, ref=ref, k=SUPPORT_K)
        self._support = {"Xs": Xs, "mean": mean, "std": std, "ref": ref, "k": SUPPORT_K}

        return {"rmse": rmse, "r2": r2}

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
                             "ref": float(d["ref"]), "k": int(d["k"])}
        return self._support

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
        """Return (lo, med, hi) arrays for a 90% prediction interval.

        Uses the quantile model when available; otherwise falls back to a flat
        ±4 MPa band around the point prediction.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        qm = self._load_quantile()
        if qm is None:
            med = self.predict_batch(X)
            return med - 4.0, med, med + 4.0
        p = np.atleast_2d(qm.predict(X))
        lo = np.minimum(p[:, 0], p[:, 2])
        hi = np.maximum(p[:, 0], p[:, 2])
        return lo, p[:, 1], hi

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

    def in_support(self, X: np.ndarray, threshold: float = 1.5) -> np.ndarray:
        """Boolean mask: True where a mix is within the trusted data region."""
        return self.novelty(X) <= threshold


if __name__ == "__main__":
    predictor = StrengthPredictor()
    predictor.train()
