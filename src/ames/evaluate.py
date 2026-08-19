import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from src.ames.config import N_FOLDS, RANDOM_SEED

def evaluate_pipeline_cv(pipeline, X, y, n_folds: int = N_FOLDS, random_seed: int = RANDOM_SEED) -> dict:
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    log_rmse_scores = []
    dollar_mae_scores = []

    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline.fit(X_train, y_train)
        preds_log = pipeline.predict(X_val)

        rmse_log = root_mean_squared_error(y_val, preds_log)
        log_rmse_scores.append(rmse_log)

        y_val_dollars = np.expm1(y_val)
        preds_dollars = np.expm1(preds_log)
        mae_dollars = mean_absolute_error(y_val_dollars, preds_dollars)
        dollar_mae_scores.append(mae_dollars)

    return {
        "log_rmse_mean": float(np.mean(log_rmse_scores)),
        "log_rmse_std": float(np.std(log_rmse_scores)),
        "dollar_mae_mean": float(np.mean(dollar_mae_scores)),
        "dollar_mae_std": float(np.std(dollar_mae_scores))
    }
