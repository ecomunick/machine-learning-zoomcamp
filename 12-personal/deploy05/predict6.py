import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# Load pipeline_v2 inside the container
with open("/code/pipeline_v2.bin", "rb") as f:
    pipeline = pickle.load(f)

app = FastAPI(title="Lead Conversion Q6")

class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

@app.post("/predict")
def predict(lead: Lead):
    lead_dict = lead.dict()
    probability = pipeline.predict_proba([lead_dict])[0, 1]
    return {"conversion_probability": probability}
