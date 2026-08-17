import numpy as np
from sklearn.linear_model import Ridge
from src.ames.data import load_raw_data, get_feature_target_split
from src.ames.pipeline import build_preprocessor, create_full_pipeline

def test_preprocessor_fit_transform():
    df = load_raw_data()
    X, y = get_feature_target_split(df)
    preprocessor = build_preprocessor(X)
    X_transformed = preprocessor.fit_transform(X)
    assert X_transformed.shape[0] == X.shape[0]
    assert not np.isnan(X_transformed).any()

def test_full_pipeline_fit_predict():
    df = load_raw_data()
    X, y = get_feature_target_split(df)
    preprocessor = build_preprocessor(X)
    pipeline = create_full_pipeline(Ridge(), preprocessor)
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert preds.shape[0] == y.shape[0]
    assert not np.isnan(preds).any()

