from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.ames.config import MODELS_DIR
from src.ames.explain import TOP_SERVING_FEATURES

DEFAULT_MODEL_PATH = MODELS_DIR / "model_v1.joblib"

_cached_model = None

def load_model(model_path: Path = DEFAULT_MODEL_PATH):
    global _cached_model
    if _cached_model is None:
        _cached_model = joblib.load(model_path)
    return _cached_model

def predict_price(data: dict | pd.DataFrame, model_path: Path = DEFAULT_MODEL_PATH):
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()

    missing_cols = [c for c in TOP_SERVING_FEATURES if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required features: {missing_cols}")

    df_subset = df[TOP_SERVING_FEATURES]

    model = load_model(model_path)
    preds_log = model.predict(df_subset)
    preds_dollars = np.expm1(preds_log)

    if len(preds_dollars) == 1:
        return float(np.round(preds_dollars[0], 2))
    return np.round(preds_dollars, 2)
