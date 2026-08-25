import pytest
import pandas as pd
from src.ames.predict import load_model, predict_price, DEFAULT_MODEL_PATH
from src.ames.train import train_and_save_final_model
from src.ames.data import load_raw_data, get_feature_target_split
from src.ames.explain import TOP_SERVING_FEATURES

@pytest.fixture(scope="session", autouse=True)
def ensure_model_artifact():
    if not DEFAULT_MODEL_PATH.exists():
        train_and_save_final_model(DEFAULT_MODEL_PATH)

def test_load_model():
    model = load_model(DEFAULT_MODEL_PATH)
    assert model is not None
    assert hasattr(model, "predict")

def test_predict_single_sample():
    sample = {
        "OverallQual": 7,
        "GrLivArea": 1710,
        "TotalBsmtSF": 856,
        "YearRemodAdd": 2003,
        "YearBuilt": 2003,
        "BsmtFinSF1": 706,
        "Fireplaces": 0,
        "OverallCond": 5,
        "GarageCars": 2,
        "LotArea": 8450,
        "MSZoning": "RL",
        "KitchenQual": "Gd"
    }
    price = predict_price(sample)
    assert isinstance(price, float)
    assert 50000 < price < 600000

def test_predict_batch():
    df = load_raw_data().head(5)
    X, _ = get_feature_target_split(df)
    preds = predict_price(X[TOP_SERVING_FEATURES])
    assert len(preds) == 5
    assert all(p > 0 for p in preds)

def test_predict_missing_feature_raises():
    invalid_sample = {
        "OverallQual": 7,
        "GrLivArea": 1710
    }
    with pytest.raises(ValueError):
        predict_price(invalid_sample)
