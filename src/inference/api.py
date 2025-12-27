from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# -----------------------------
# App
# -----------------------------
app = FastAPI(title="Fraud Detection API")

# -----------------------------
# Load model (dummy au début)
# -----------------------------
MODEL_PATH = "models/model.joblib"

try:
    model = joblib.load(MODEL_PATH)
except Exception:
    model = None  # permet de lancer l'API même sans modèle

# -----------------------------
# Pydantic schema
# -----------------------------
class TransactionIn(BaseModel):
    TransactionAmt: float
    ProductCD: str
    card1: int
    addr1: float | None = None

class PredictionOut(BaseModel):
    fraud_probability: float

# -----------------------------
# Health check (OBLIGATOIRE pour K8s)
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------------
# Predict endpoint
# -----------------------------
@app.post("/predict", response_model=PredictionOut)
def predict(payload: TransactionIn):
    if model is None:
        return {"fraud_probability": 0.0}

    df = pd.DataFrame([payload.dict()])
    df["ProductCD"] = df["ProductCD"].astype("category").cat.codes
    df["addr1"] = df["addr1"].fillna(0)

    proba = float(model.predict_proba(df)[0][1])
    return {"fraud_probability": proba}
