
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from xgboost import XGBClassifier

app = FastAPI(title="Fraud API")

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/artifacts/model_xgb.bin"))
_model: XGBClassifier | None = None


class PredictRequest(BaseModel):
    records: List[Dict[str, Any]]


def _load_model() -> XGBClassifier:
    global _model
    if _model is not None:
        return _model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))
    _model = model
    return _model


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, List[float]]:
    if not req.records:
        raise HTTPException(status_code=400, detail="No records provided")
    try:
        model = _load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    df = pd.DataFrame(req.records)
    proba = model.predict_proba(df)[:, 1]
    return {"predictions": proba.tolist()}

