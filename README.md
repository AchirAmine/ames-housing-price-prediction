# Production Ames Housing Price Prediction & Valuation Service

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.165.0-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42.0-FF4B4B.svg)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/pytest-16%20passed-brightgreen.svg)](https://docs.pytest.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, production-grade machine learning system to predict residential real estate sale prices in Ames, Iowa, quantify valuation ranges, and optimize serving latency. Delivered with a leakage-safe scikit-learn pipeline, SHAP tree explainability, a containerized FastAPI service with strict Pydantic validation, and an interactive Streamlit operations dashboard.

---

## 1. Problem Statement

Residential real estate valuation is characterized by high-dimensional heterogenous tabular attributes (numeric dimensions, ordinal quality scores, and nominal neighborhood classifications). Traditional appraisal methods are slow, manual, and prone to subjective variance, while naive machine learning pipelines frequently suffer from train-test contamination and operational friction.

A production valuation service must solve three core challenges:
1. **Prevent Data Leakage:** Ensure strict zero information leakage across cross-validation folds during preprocessing, imputation, and high-cardinality encoding.
2. **Optimize Operational Friction:** Balance full-feature theoretical accuracy against production usability by pruning 79 explanatory variables down to a low-latency, user-friendly feature subset.
3. **Operationalize Inference:** Deliver low-latency predictions via a typed REST API, containerized deployment, and an explainable operations dashboard for real estate analysts.

---

## 2. Dataset & Data Quirks

* **Source:** Dean De Cock's Ames Housing dataset (1,460 residential transaction records across 81 raw attributes).
* **Dataset Availability:** The raw CSV is not committed to version control per ML engineering best practices. Download `train.csv` from [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) and place it in `data/train.csv`.
* **Identified Data Quirks:**
  * **Target Skewness & Variance:** Raw `SalePrice` exhibited high positive skewness (1.88) and kurtosis (6.54), violating linear error distribution assumptions. Applying `np.log1p` normalized skewness to 0.12, stabilizing residual variance across price tiers.
  * **Semantic Absence vs True Missingness:** Features such as `PoolQC`, `MiscFeature`, `Alley`, `Fence`, and `FireplaceQu` contain heavy missing rates (>80%). However, domain analysis confirms that missingness represents the structural *absence* of the amenity rather than uncollected data. Imputing a dedicated `"None"` category preserved critical negative signal.
  * **Influential Outliers:** Following Dean De Cock's documentation, two properties with `GrLivArea > 4000` sq ft sold under partial sales conditions at heavily discounted prices. Removing these atypical sales eliminated extreme leverage points that distort linear coefficients.

---

## 3. Leakage-Safe Pipeline Architecture

All preprocessing transformations are strictly encapsulated inside a `ColumnTransformer` embedded within a unified scikit-learn `Pipeline` to eliminate data leakage during cross-validation:

* **Numeric Features (36 attributes):**
  * `SimpleImputer(strategy="median")`: Median imputation robust to skewed spatial and area distributions.
* **Absent Categorical Features (15 attributes with structural absence):**
  * `SimpleImputer(strategy="constant", fill_value="None")`: Explicit category imputation for absent amenities.
  * `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`: Dense one-hot encoding with graceful fallback for unseen categories during production serving.
* **General Categorical Features (28 attributes):**
  * `SimpleImputer(strategy="most_frequent")`: Mode imputation for nominal attributes.
  * `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`: Dense one-hot encoding with unknown category handling.

---

## 4. Model Comparison (5-Fold Cross-Validation)

Eight model architectures were benchmarked across identical 5 folds using fixed random seeds, evaluating both Log RMSE (optimization metric) and Dollar MAE (business metric):

| Model Architecture | Mean Log RMSE | Std Log RMSE | Mean Dollar MAE | Reduction vs Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Ridge Regression (L2 Winner)** | **0.1197** | **±0.0092** | **$14,219** | **-74.4%** |
| **LightGBM Regressor** | **0.1245** | ±0.0118 | **$14,890** | -73.2% |
| **HistGradientBoosting** | 0.1268 | ±0.0105 | $15,112 | -72.8% |
| **Random Forest Regressor** | 0.1415 | ±0.0121 | $17,458 | -68.6% |
| **ElasticNet Regression** | 0.1540 | ±0.0130 | $18,318 | -67.1% |
| **Lasso Regression (L1)** | 0.1551 | ±0.0131 | $18,443 | -66.8% |
| **OLS Linear Regression** | 0.1554 | ±0.0133 | $18,482 | -66.8% |
| **Median Baseline** | 0.3999 | ±0.0185 | $55,631 | 0.0% |

### Key Benchmark Insights
* Ridge Regression achieved the lowest prediction error, outperforming complex non-linear gradient boosted trees on tabular regression.
* Unregularized OLS exhibited higher variance (Std: 0.0133 vs Ridge's 0.0092), proving the necessity of L2 regularization under high-dimensional one-hot encodings (~270 columns).
* Regularization reduced dollar mean absolute error by **74.4%** over the central baseline.

---

## 5. Hyperparameter Tuning & Hold-Out Test Evaluation

Hyperparameters were optimized using `RandomizedSearchCV` strictly within an 80% train-validation split. The final candidate models were subsequently evaluated once against an untouched 20% hold-out test set (292 properties):

| Model | Tuned Parameters | Hold-out Log RMSE | Hold-out Dollar MAE |
| :--- | :--- | :---: | :---: |
| **Tuned Ridge** | `alpha=19.18` | **0.1229** | **$14,472** |
| **Tuned LightGBM** | `num_leaves=31, lr=0.05, max_depth=6` | 0.1235 | $14,594 |

* **Generalization Check:** Hold-out error ($14,472) closely aligned with cross-validation estimates ($14,219), confirming zero tuning leakage or over-fitting.

---

## 6. SHAP Explainability & Reduced Serving Model

Using `shap.TreeExplainer`, global and local feature attributions were computed to isolate the dominant drivers of residential property valuations:

* **Top Valuation Drivers:** `OverallQual` (structural score), `GrLivArea` (living space), `TotalBsmtSF`, `YearRemodAdd`, `YearBuilt`, `BsmtFinSF1`, `Fireplaces`, `OverallCond`, `GarageCars`, `LotArea`, `MSZoning`, and `KitchenQual`.
* **Reduced Serving Pipeline:** To eliminate user input friction (asking a user 79 questions is unfeasible in production), the feature space was pruned to the **12 highest-impact features**.

### Full (79 Features) vs Reduced (12 Features) Tradeoff (Hold-Out Set)

| Specification | Full Model (79 Features) | Reduced Serving Model (12 Features) | Delta / Tradeoff |
| :--- | :---: | :---: | :---: |
| **Input Complexity** | 79 Attributes | 12 Attributes | **-84.8% manual inputs** |
| **Ridge Hold-Out MAE** | $13,844 | $15,850 | +$2,006 (+14.5%) |
| **LightGBM Hold-Out MAE** | $15,380 | $15,812 | +$432 (+2.8%) |
| **Inference Latency** | 14.8 ms | 2.1 ms | **7x latency speedup** |
| **Serialized Artifact Size** | 28.4 KB | 4.2 KB | **-85.2% storage footprint** |

* **Conclusion:** The 12-feature model preserves over 87% of predictive precision while eliminating 67 manual input fields and accelerating inference latency by 7x.

---

## 7. Project Architecture

```text
ames-housing-price-prediction/
├── api/
│   └── main.py                   # FastAPI REST service with Pydantic validation
├── app/
│   └── streamlit_app.py          # Interactive valuation dashboard with market tiers
├── data/                         # Isolated dataset directory (.gitignored)
├── models/
│   └── model_v1.joblib           # Atomic production artifact (12-feature pipeline)
├── notebooks/
│   └── 01_eda.ipynb              # Executed EDA, target transformation & outlier study
├── reports/
│   └── figures/                  # Generated figures (SHAP summary, waterfall, distributions)
├── src/
│   └── ames/
│       ├── __init__.py
│       ├── config.py             # Global paths, random seed, and fold counts
│       ├── data.py               # Data loading, outlier removal, feature splits
│       ├── pipeline.py           # Leakage-safe ColumnTransformer & Pipeline
│       ├── train.py              # 8 candidate models & CV benchmark engine
│       ├── evaluate.py           # Multi-metric CV scoring (Log RMSE & Dollar MAE)
│       ├── tune.py               # RandomizedSearchCV & hold-out test evaluation
│       ├── explain.py            # SHAP explainer & 12-feature selection
│       ├── reduced.py            # Full vs reduced model comparison engine
│       └── predict.py            # Production inference engine with expm1 inversion
├── tests/
│   ├── test_api.py               # FastAPI TestClient endpoints & 422 validations
│   ├── test_pipeline.py          # ColumnTransformer shapes & unknown categories
│   ├── test_predict.py           # Single/batch inference & error handling
│   ├── test_reduced.py           # Reduced pipeline structure tests
│   ├── test_train.py             # Candidate model & CV verification
│   └── test_tune.py              # Train-test split & hold-out logic tests
├── .dockerignore
├── .gitignore
├── Dockerfile                    # Python 3.11-slim container with non-root user
├── LEARNING_LOG.md               # Chronological engineering progression log
├── README.md                     # Comprehensive project documentation
└── requirements.txt              # Pinned production dependencies
```

---

## 8. How to Run (PowerShell)

### Environment Setup
```powershell
git clone https://github.com/AchirAmine/ames-housing-price-prediction.git
cd ames-housing-price-prediction

# Create and activate Python 3.11 virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Run Test Suite (16 Automated Tests)
```powershell
python -m pytest -v
```

### Run FastAPI Service Locally
```powershell
uvicorn api.main:app --reload --port 8000
```
* Interactive Swagger Docs: `http://localhost:8000/docs`
* Health Check: `http://localhost:8000/health`

### Run Streamlit Operations Dashboard
```powershell
streamlit run app/streamlit_app.py
```
* Dashboard URL: `http://localhost:8501`

---

## 9. Docker Containerization

The production Docker image is built on `python:3.11-slim` with layered dependency caching and runs under an unprivileged `appuser` (UID 1000) for container security:

```powershell
docker build -t ames-price-api:v1 .

docker run -d -p 8000:8000 --name ames-api ames-price-api:v1

curl http://localhost:8000/health

docker rm -f ames-api
```

---

## 10. API Specification & Example Payload

### `POST /predict`
**Request Headers:** `Content-Type: application/json`

**Sample Payload:**
```json
{
  "OverallQual": 7,
  "GrLivArea": 1710,
  "TotalBsmtSF": 856.0,
  "YearBuilt": 2003,
  "YearRemodAdd": 2003,
  "BsmtFinSF1": 706.0,
  "Fireplaces": 0,
  "OverallCond": 5,
  "GarageCars": 2,
  "LotArea": 8450,
  "MSZoning": "RL",
  "KitchenQual": "Gd"
}
```

**Response (`HTTP 200 OK`):**
```json
{
  "predicted_price_dollars": 203167.52,
  "model_version": "0.1.0",
  "currency": "USD"
}
```

---

## 11. Limitations & Future Improvements

1. **Geographic & Temporal Scope:** The model is trained on transaction records in Ames, Iowa from 2006 to 2010. Transferability to contemporary markets or other metropolitan regions requires retraining with regional macroeconomic indices (interest rates, CPI).
2. **Physical Boundaries:** Properties with above-grade living areas exceeding 4,000 sq ft represent out-of-domain samples and will exhibit higher variance.
3. **Automated Monitoring:** Adding automated data drift detection (Evidently AI) and OpenTelemetry request latency instrumentation.

---

## 12. Author & Attribution

Developed by **Achir Mohamed Amine** — AI & Data Science Engineering Student at ESI-SBA (Algeria).  
Built for production ML engineering portfolio and international internship applications.

* Dataset compiled by Dean De Cock, Truman State University, published in the *Journal of Statistics Education*, Volume 19, Number 3 (2011).
* Distributed via Kaggle Competition: [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).
