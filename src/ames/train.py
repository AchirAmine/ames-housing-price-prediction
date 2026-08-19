import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from lightgbm import LGBMRegressor

from src.ames.config import RANDOM_SEED
from src.ames.data import load_raw_data, remove_known_outliers, get_feature_target_split
from src.ames.pipeline import build_preprocessor, create_full_pipeline
from src.ames.evaluate import evaluate_pipeline_cv

def get_candidate_models():
    return {
        "Baseline (Median)": DummyRegressor(strategy="median"),
        "Linear Regression (OLS)": LinearRegression(),
        "Ridge": Ridge(random_state=RANDOM_SEED),
        "Lasso": Lasso(random_state=RANDOM_SEED, max_iter=3000),
        "ElasticNet": ElasticNet(random_state=RANDOM_SEED, max_iter=3000),
        "Random Forest": RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=RANDOM_SEED),
        "LightGBM": LGBMRegressor(random_state=RANDOM_SEED, verbose=-1, n_jobs=-1)
    }

def run_model_comparison(drop_outliers: bool = True) -> pd.DataFrame:
    df = load_raw_data()
    if drop_outliers:
        df = remove_known_outliers(df)

    X, y = get_feature_target_split(df)
    preprocessor = build_preprocessor(X)
    models = get_candidate_models()

    records = []
    for name, model in models.items():
        pipeline = create_full_pipeline(model, preprocessor)
        metrics = evaluate_pipeline_cv(pipeline, X, y)
        records.append({
            "Model": name,
            "Log RMSE (Mean)": metrics["log_rmse_mean"],
            "Log RMSE (Std)": metrics["log_rmse_std"],
            "Dollar MAE (Mean)": metrics["dollar_mae_mean"],
            "Dollar MAE (Std)": metrics["dollar_mae_std"]
        })

    results_df = pd.DataFrame(records).sort_values(by="Log RMSE (Mean)")
    return results_df

if __name__ == "__main__":
    results = run_model_comparison(drop_outliers=True)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print("\n=== Ames Housing 5-Fold Cross-Validation Comparison ===")
    print(results.to_string(index=False))
