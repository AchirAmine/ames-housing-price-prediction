import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

valid_payload = {
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

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_version": "0.1.0"}

def test_predict_endpoint_valid():
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price_dollars" in data
    assert data["predicted_price_dollars"] > 50000
    assert data["model_version"] == "0.1.0"
    assert data["currency"] == "USD"

def test_predict_endpoint_invalid_overall_qual():
    payload = valid_payload.copy()
    payload["OverallQual"] = 0
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_endpoint_missing_field():
    payload = valid_payload.copy()
    del payload["GrLivArea"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
