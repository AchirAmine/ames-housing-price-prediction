# Learning Log — Ames Housing Price Prediction

## S1 — Project Setup (2026-08-11)

**What I did:** Created the production src layout, configured a dedicated Python 3.11 virtual environment with all required dependencies, set up .gitignore to isolate datasets and model artifacts, connected the remote GitHub repository, and documented the project foundation.

**What I learned:** Isolating dependencies in a dedicated virtual environment prevents package version conflicts. A clean src package layout separates reusable production code from notebooks and deployment interfaces, while .gitignore safeguards repository performance and data hygiene.

## S2 — Exploratory Data Analysis & Target Transformation (2026-08-14)

**What I did:** Analyzed the 1,460 residential transaction records across 81 attributes, resolved target distribution skewness (1.88) via log1p normalization, distinguished semantic absence from true missingness in amenities (PoolQC, MiscFeature, Alley), and identified Dean De Cock's documented physical outliers (GrLivArea > 4000) in an executed notebook.

**What I learned:** Target distribution skewness dropped from 1.88 to 0.12 using log1p, stabilizing linear error variance. Imputing a dedicated 'None' category for absent amenities preserves critical negative structural signal, whereas discarding or blindly mode-imputing would introduce severe domain distortion.

## S3 — Leakage-Safe ColumnTransformer & Preprocessing Pipeline (2026-08-17)

**What I did:** Built a modular ColumnTransformer encapsulating median imputation for 36 numeric features alongside dedicated 'None' constant imputation for 15 absent-categorical features and mode imputation for 28 general categoricals, integrated into an end-to-end Pipeline validated by automated unit tests.

**What I learned:** Encapsulating feature transformations strictly inside the Pipeline ensures that imputers and encoders are fitted exclusively on training folds during cross-validation, completely eliminating data leakage. Setting handle_unknown="ignore" on OneHotEncoder ensures robust inference when unseen categories arrive at serving time.

## S4 — 5-Fold Cross-Validation Model Comparison (2026-08-19)

**What I did:** Benchmarked 8 candidate models (Median Baseline, OLS, Lasso, ElasticNet, Random Forest, HistGradientBoosting, LightGBM, Ridge) across 5 folds measuring both Log RMSE (optimization metric) and Dollar MAE (business metric).

**What I learned:** Ridge Regression with L2 regularization achieved the lowest prediction error (Log RMSE: 0.1197, MAE: $14,219), reducing dollar error by 74.4% compared to the baseline ($55,631) and outperforming complex tree ensembles. Unregularized OLS exhibited higher variance (Std: 0.0133 vs Ridge's 0.0092), proving the necessity of L2 shrinkage under high-dimensional one-hot encodings.

## S5 — Hyperparameter Tuning & Hold-Out Test Evaluation (2026-08-21)

**What I did:** Optimized hyperparameters via RandomizedSearchCV strictly on an 80% train-validation split, tuned Ridge alpha to 19.18, and evaluated candidate generalizability on an untouched 20% hold-out test set (292 properties).

**What I learned:** Tuned Ridge achieved Log RMSE 0.1229 ($14,472 MAE) on the unseen hold-out set, closely matching cross-validation estimates ($14,219 MAE) and outperforming Tuned LightGBM ($14,594 MAE). This confirms genuine out-of-sample generalizability without hyperparameter over-fitting or validation leakage.
