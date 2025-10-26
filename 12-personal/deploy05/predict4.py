# from fastapi import FastAPI
# import pickle
# from pydantic import BaseModel
# import uvicorn  # <- we need to import uvicorn for the main block

# # Load the pipeline
# with open("pipeline_v1.bin", "rb") as f:
#     pipeline = pickle.load(f)

# # Create FastAPI app
# app = FastAPI(title="Lead Scoring API")

# # Define input data model
# class Lead(BaseModel):
#     lead_source: str
#     number_of_courses_viewed: int
#     annual_income: float

# # Define endpoint
# @app.post("/predict")
# def predict(lead: Lead):
#     # Convert to dict
#     data = lead.model_dump() # for pydantic v2 # lead.dict()
#     # Predict probability
#     prob = pipeline.predict_proba([data])[0, 1]
#     return {"conversion_probability": prob}

# # ----------------------------
# # Run the server when executed directly
# # ----------------------------
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=9696)


# predict4.py

import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Load the pipeline
with open("pipeline_v1.bin", "rb") as f:
    model = pickle.load(f)

# 2. Define the input schema (for request validation)
class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

# 3. Create FastAPI app
app = FastAPI(title="Lead Conversion Prediction API")

# 4. Define the prediction endpoint
@app.post("/predict")
def predict(lead: Lead):
    data = lead.model_dump()
    proba = model.predict_proba([data])[0, 1]
    return {"conversion_probability": proba}
