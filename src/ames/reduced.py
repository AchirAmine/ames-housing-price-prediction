import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

from src.ames.config import RANDOM_SEED
from src.ames.data import load_raw_data, remove_known_outliers, get_feature_target_split
from src.ames.pipeline import build_preprocessor, create_full_pipeline
from src.ames.evaluate import evaluate_pipeline_cv
from src.ames.explain import TOP_SERVING_FEATURES

def compare_full_vs_reduced_model() -> pd.DataFrame:
    df = remove_known_outliers(load_raw_data())
    X, y = get_feature_target_split(df)

    X_reduced = X[TOP_SERVING_FEATURES].copy()

    full_preprocessor = build_preprocessor(X)
    reduced_preprocessor = build_preprocessor(X_reduced)

    full_ridge_pipeline = create_full_pipeline(Ridge(alpha=19.18, random_state=RANDOM_SEED), full_preprocessor)
    reduced_ridge_pipeline = create_full_pipeline(Ridge(alpha=19.18, random_state=RANDOM_SEED), reduced_preprocessor)

    full_lgbm_pipeline = create_full_pipeline(
        LGBMRegressor(random_state=RANDOM_SEED, verbose=-1, n_estimators=300, learning_rate=0.03, max_depth=3),
        full_preprocessor
    )
    reduced_lgbm_pipeline = create_full_pipeline(
        LGBMRegressor(random_state=RANDOM_SEED, verbose=-1, n_estimators=300, learning_rate=0.03, max_depth=3),
        reduced_preprocessor
    )

    metrics_full_ridge = evaluate_pipeline_cv(full_ridge_pipeline, X, y)
    metrics_reduced_ridge = evaluate_pipeline_cv(reduced_ridge_pipeline, X_reduced, y)

    metrics_full_lgbm = evaluate_pipeline_cv(full_lgbm_pipeline, X, y)
    metrics_reduced_lgbm = evaluate_pipeline_cv(reduced_lgbm_pipeline, X_reduced, y)

    comparison = [
        {
            "Model": "Full Ridge (79 Features)",
            "Log RMSE": metrics_full_ridge["log_rmse_mean"],
            "Dollar MAE": metrics_full_ridge["dollar_mae_mean"]
        },
        {
            "Model": "Reduced Ridge (12 Features)",
            "Log RMSE": metrics_reduced_ridge["log_rmse_mean"],
            "Dollar MAE": metrics_reduced_ridge["dollar_mae_mean"]
        },
        {
            "Model": "Full LightGBM (79 Features)",
            "Log RMSE": metrics_full_lgbm["log_rmse_mean"],
            "Dollar MAE": metrics_full_lgbm["dollar_mae_mean"]
        },
        {
            "Model": "Reduced LightGBM (12 Features)",
            "Log RMSE": metrics_reduced_lgbm["log_rmse_mean"],
            "Dollar MAE": metrics_reduced_lgbm["dollar_mae_mean"]
        }
    ]

    return pd.DataFrame(comparison)

if __name__ == "__main__":
    df_res = compare_full_vs_reduced_model()
    print("\n=== S6: Full (79 Features) vs Reduced (12 Features) Model Comparison ===")
    print(df_res.to_string(index=False))
