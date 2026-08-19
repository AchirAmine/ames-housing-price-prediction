from src.ames.train import get_candidate_models
from src.ames.evaluate import evaluate_pipeline_cv
from src.ames.pipeline import build_preprocessor, create_full_pipeline
from src.ames.data import load_raw_data, get_feature_target_split

def test_candidate_models_presence():
    models = get_candidate_models()
    assert "Baseline (Median)" in models
    assert "Ridge" in models
    assert "LightGBM" in models

def test_evaluate_pipeline_cv_output():
    df = load_raw_data().head(100)
    X, y = get_feature_target_split(df)
    preprocessor = build_preprocessor(X)
    models = get_candidate_models()
    pipeline = create_full_pipeline(models["Ridge"], preprocessor)
    res = evaluate_pipeline_cv(pipeline, X, y, n_folds=2)
    assert "log_rmse_mean" in res
    assert "dollar_mae_mean" in res
    assert res["log_rmse_mean"] > 0
    assert res["dollar_mae_mean"] > 0
