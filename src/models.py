import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from .data_fetcher import load_data
import os

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
STRENGTH_MODEL_PATH = os.path.join(MODEL_DIR, "strength_model.json")

class StrengthPredictor:
    def __init__(self):
        self.model = None
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

    def train(self):
        """Trains an XGBoost model on the UIUC dataset."""
        df = load_data()
        X = df.drop("strength", axis=1)
        y = df["strength"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)

        # Evaluation
        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Trained. RMSE: {rmse:.2f}, R2: {r2:.2f}")
        self.model.save_model(STRENGTH_MODEL_PATH)
        return {"rmse": rmse, "r2": r2}

    def predict(self, mix_design: np.ndarray) -> float:
        """Predicts strength for a given mix design (numpy array)."""
        if self.model is None:
            if os.path.exists(STRENGTH_MODEL_PATH):
                self.model = xgb.XGBRegressor()
                self.model.load_model(STRENGTH_MODEL_PATH)
            else:
                raise ValueError("Model not trained. Run train() first.")
        
        # Ensure input is 2D
        if mix_design.ndim == 1:
            mix_design = mix_design.reshape(1, -1)
            
        return float(self.model.predict(mix_design)[0])

if __name__ == "__main__":
    predictor = StrengthPredictor()
    predictor.train()
