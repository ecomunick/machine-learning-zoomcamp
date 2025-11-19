from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

class DiabetesInput(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.post("/predict")
def predict_diabetes(data: DiabetesInput):
    X = np.array([[v for v in data.dict().values()]])
    prob = model.predict_proba(X)[0,1]
    return {"diabetes_probability": float(prob)}
