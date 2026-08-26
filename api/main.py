from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from src.ames.predict import predict_price, DEFAULT_MODEL_PATH

MODEL_VERSION = "0.1.0"

app = FastAPI(
    title="Ames Housing Price Prediction API",
    version=MODEL_VERSION
)

class HouseFeatures(BaseModel):
    OverallQual: int = Field(..., ge=1, le=10)
    GrLivArea: int = Field(..., ge=100, le=8000)
    TotalBsmtSF: float = Field(..., ge=0, le=6000)
    YearBuilt: int = Field(..., ge=1850, le=2026)
    YearRemodAdd: int = Field(..., ge=1850, le=2026)
    BsmtFinSF1: float = Field(..., ge=0, le=5000)
    Fireplaces: int = Field(..., ge=0, le=5)
    OverallCond: int = Field(..., ge=1, le=10)
    GarageCars: int = Field(..., ge=0, le=5)
    LotArea: int = Field(..., ge=1000, le=200000)
    MSZoning: Literal["RL", "RM", "C (all)", "FV", "RH"]
    KitchenQual: Literal["Ex", "Gd", "TA", "Fa", "Po"]

class PredictionResponse(BaseModel):
    predicted_price_dollars: float
    model_version: str
    currency: str

class HealthResponse(BaseModel):
    status: str
    model_version: str

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="healthy", model_version=MODEL_VERSION)

@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    try:
        price = predict_price(features.model_dump(), model_path=DEFAULT_MODEL_PATH)
        return PredictionResponse(
            predicted_price_dollars=price,
            model_version=MODEL_VERSION,
            currency="USD"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
