# Ames Housing Price Prediction

> End-to-end ML service: leakage-safe pipeline · model comparison · REST API · Docker · Streamlit demo

---

## Problem

Predict residential home sale prices in Ames, Iowa using 79 explanatory variables.
This is a regression task on the [Kaggle House Prices dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

## Data

Download `train.csv` from [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
and place it in `data/train.csv`. The dataset is provided under Kaggle competition rules and
is **not included** in this repository.

## Approach

_To be completed after each step._

## Results

_Real numbers from my own runs — to be filled in after S5._

## How to Run

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Limitations

- Data covers Ames, Iowa residential sales from 2006–2010 only.
- Not valid for other cities, countries, or today's market conditions.
- This is a portfolio/learning project, not a valuation tool.

## Project Structure

```
ames/
├── notebooks/          # EDA only
├── src/ames/           # Core library (pipeline, training, evaluation)
├── api/                # FastAPI service
├── app/                # Streamlit demo
├── tests/              # pytest tests
├── models/             # Saved model artifact (small final model only)
├── reports/figures/    # Plots and charts
├── data/               # NOT committed — download from Kaggle
├── requirements.txt
├── Dockerfile
└── LEARNING_LOG.md
```

## Attribution

Dataset: Dean De Cock, "Ames, Iowa: Alternative to the Boston Housing Data Set", 2011.
Kaggle competition: House Prices - Advanced Regression Techniques.
