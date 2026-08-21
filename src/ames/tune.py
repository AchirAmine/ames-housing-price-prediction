import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

from src.ames.config import RANDOM_SEED, N_FOLDS
from src.ames.data import load_raw_data, remove_known_outliers, get_feature_target_split
from src.ames.pipeline import build_preprocessor, create_full_pipeline

def get_train_test_split(test_size: float = 0.2, random_seed: int = RANDOM_SEED, drop_outliers: bool = True):
    df = load_raw_data()
    if drop_outliers:
        df = remove_known_outliers(df)
    X, y = get_feature_target_split(df)
    return train_test_split(X, y, test_size=test_size, random_state=random_seed)

def tune_ridge(X_train, y_train, n_iter: int = 50, random_seed: int = RANDOM_SEED):
    preprocessor = build_preprocessor(X_train)
    pipeline = create_full_pipeline(Ridge(random_state=random_seed), preprocessor)

    param_distributions = {
        "model__alpha": np.logspace(-2, 3, 100)
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=N_FOLDS,
        scoring="neg_root_mean_squared_error",
        random_state=random_seed,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search

def tune_lightgbm(X_train, y_train, n_iter: int = 30, random_seed: int = RANDOM_SEED):
    preprocessor = build_preprocessor(X_train)
    pipeline = create_full_pipeline(
        LGBMRegressor(random_state=random_seed, verbose=-1, n_jobs=-1),
        preprocessor
    )

    param_distributions = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__num_leaves": [15, 31, 63],
        "model__max_depth": [-1, 3, 5, 8],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0],
        "model__reg_alpha": [0.0, 0.1, 1.0, 5.0],
        "model__reg_lambda": [0.0, 0.1, 1.0, 5.0]
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=N_FOLDS,
        scoring="neg_root_mean_squared_error",
        random_state=random_seed,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search

def evaluate_final_holdout(best_pipeline, X_train, y_train, X_test, y_test) -> dict:
    best_pipeline.fit(X_train, y_train)
    preds_log = best_pipeline.predict(X_test)

    log_rmse = root_mean_squared_error(y_test, preds_log)
    y_test_dollars = np.expm1(y_test)
    preds_dollars = np.expm1(preds_log)
    dollar_mae = mean_absolute_error(y_test_dollars, preds_dollars)

    return {
        "log_rmse": float(log_rmse),
        "dollar_mae": float(dollar_mae)
    }

def run_tuning_and_final_evaluation():
    X_train, X_test, y_train, y_test = get_train_test_split(test_size=0.2, random_seed=RANDOM_SEED)

    print("Running Ridge Hyperparameter Tuning...")
    ridge_search = tune_ridge(X_train, y_train)
    best_ridge = ridge_search.best_estimator_
    ridge_cv_rmse = -ridge_search.best_score_
    ridge_holdout = evaluate_final_holdout(best_ridge, X_train, y_train, X_test, y_test)

    print("Running LightGBM Hyperparameter Tuning...")
    lgbm_search = tune_lightgbm(X_train, y_train)
    best_lgbm = lgbm_search.best_estimator_
    lgbm_cv_rmse = -lgbm_search.best_score_
    lgbm_holdout = evaluate_final_holdout(best_lgbm, X_train, y_train, X_test, y_test)

    summary = [
        {
            "Model": "Tuned Ridge",
            "Best CV Log RMSE": ridge_cv_rmse,
            "Hold-out Log RMSE": ridge_holdout["log_rmse"],
            "Hold-out Dollar MAE": ridge_holdout["dollar_mae"],
            "Best Params": ridge_search.best_params_
        },
        {
            "Model": "Tuned LightGBM",
            "Best CV Log RMSE": lgbm_cv_rmse,
            "Hold-out Log RMSE": lgbm_holdout["log_rmse"],
            "Hold-out Dollar MAE": lgbm_holdout["dollar_mae"],
            "Best Params": lgbm_search.best_params_
        }
    ]

    return summary

if __name__ == "__main__":
    results = run_tuning_and_final_evaluation()
    print("\n=== S5: Final Hold-out Evaluation Results ===")
    for r in results:
        print(f"\nModel: {r['Model']}")
        print(f"  Best 5-Fold CV Log RMSE: {r['Best CV Log RMSE']:.4f}")
        print(f"  Hold-out Log RMSE:       {r['Hold-out Log RMSE']:.4f}")
        print(f"  Hold-out Dollar MAE:     ${r['Hold-out Dollar MAE']:,.2f}")
        print(f"  Best Parameters:         {r['Best Params']}")
