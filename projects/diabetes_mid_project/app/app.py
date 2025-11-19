# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path
import os

# ------------------------
# Load model
# ------------------------
# MODEL_PATH = Path('../models/xgboost_model.pkl') 
# model = joblib.load(MODEL_PATH)

# get the directory of the current file (app.py)
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR.parent / 'models' / 'xgboost_model.pkl'
model = joblib.load(MODEL_PATH)

# ------------------------
# FastAPI app
# ------------------------
app = FastAPI(title="Diabetes Prediction API", version="1.0")

# ------------------------
# Request schema
# ------------------------
class PatientData(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# ------------------------
# Prediction endpoint
# ------------------------
@app.post("/predict")
def predict(patient: PatientData):
    # Convert input to DataFrame
    df = pd.DataFrame([patient.dict()])
    # Predict
    prob = model.predict_proba(df)[:, 1][0]
    pred = int(model.predict(df)[0])
    return {"predicted_class": pred, "predicted_proba": float(prob)}

# ------------------------
# Health check endpoint
# ------------------------
@app.get("/")
def health():
    return {"status": "ok"}
