from src.ames.explain import TOP_SERVING_FEATURES
from src.ames.data import load_raw_data, get_feature_target_split
from src.ames.pipeline import build_preprocessor, create_full_pipeline
from sklearn.linear_model import Ridge
import numpy as np

def test_top_serving_features_count():
    assert len(TOP_SERVING_FEATURES) == 12
    assert "OverallQual" in TOP_SERVING_FEATURES
    assert "GrLivArea" in TOP_SERVING_FEATURES

def test_reduced_pipeline_fit_predict():
    df = load_raw_data().head(100)
    X, y = get_feature_target_split(df)
    X_reduced = X[TOP_SERVING_FEATURES].copy()
    preprocessor = build_preprocessor(X_reduced)
    pipeline = create_full_pipeline(Ridge(), preprocessor)
    pipeline.fit(X_reduced, y)
    preds = pipeline.predict(X_reduced)
    assert preds.shape[0] == y.shape[0]
    assert not np.isnan(preds).any()
