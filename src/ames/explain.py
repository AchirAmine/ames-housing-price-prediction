import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor

from src.ames.config import FIGURES_DIR, RANDOM_SEED
from src.ames.data import load_raw_data, remove_known_outliers, get_feature_target_split
from src.ames.pipeline import build_preprocessor

TOP_SERVING_FEATURES = [
    "OverallQual",
    "GrLivArea",
    "TotalBsmtSF",
    "YearRemodAdd",
    "YearBuilt",
    "BsmtFinSF1",
    "Fireplaces",
    "OverallCond",
    "GarageCars",
    "LotArea",
    "MSZoning",
    "KitchenQual"
]

def generate_shap_plots(save_plots: bool = True):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = remove_known_outliers(load_raw_data())
    X, y = get_feature_target_split(df)

    preprocessor = build_preprocessor(X)
    X_trans = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()

    model = LGBMRegressor(
        random_state=RANDOM_SEED,
        verbose=-1,
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.8
    )
    model.fit(X_trans, y)

    explainer = shap.TreeExplainer(model)
    shap_explanation = explainer(X_trans)
    shap_explanation.feature_names = list(feature_names)

    if save_plots:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_explanation.values, X_trans, feature_names=feature_names, max_display=15, show=False)
        plt.title("SHAP Global Feature Importance (Top 15 Features)", fontsize=12)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 6))
        sample_idx = 0
        shap.plots.waterfall(shap_explanation[sample_idx], max_display=12, show=False)
        plt.title(f"SHAP Local Explanation (House Sample #{sample_idx})", fontsize=12)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "shap_waterfall.png", dpi=300, bbox_inches="tight")
        plt.close()

    return explainer, shap_explanation

if __name__ == "__main__":
    generate_shap_plots(save_plots=True)
    print("SHAP plots generated and saved successfully in reports/figures/")
