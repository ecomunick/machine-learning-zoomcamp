## Diabetes Prediction API

Machine Learning Zoomcamp – Midterm Project
FastAPI • XGBoost • Docker • Deployment

#### 🔍 Project Summary

This project predicts whether a patient is likely to have diabetes using the **PIMA Indians Diabetes Dataset**.
It includes:
- Data exploration & preprocessing
- Model training using XGBoost
- Saving the model using joblib
- Serving predictions through a FastAPI web service
- Running locally with Uvicorn
- Containerization with Docker
- Optional deployment to the cloud

### 📊 Dataset

PIMA Diabetes Dataset (UCI/Kaggle):

- 768 samples
- 8 numerical features (glucose, BMI, age, etc.)
- Binary target: diabetes (1) / no diabetes (0)

Dataset source:
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

### ⚙️ Training
The notebook in `notebooks/train.ipynb` trains an XGBoost model and exports:
`models/xgboost_model.pkl`

### 🚀 API Usage
Run locally (no Docker):
`uvicorn app.app:app --reload --port 8000`

Open the interactive API docs:
👉 http://127.0.0.1:8000/docs

Example JSON input:
`
{
  "Pregnancies": 2,
  "Glucose": 134,
  "BloodPressure": 70,
  "SkinThickness": 25,
  "Insulin": 100,
  "BMI": 28.5,
  "DiabetesPedigreeFunction": 0.3,
  "Age": 45
}
`

### 🐳 Docker
Build the image:
`docker build -t diabetes-api .`

Run the container:
`docker run -p 8000:8000 diabetes-api`

Open in browser:
http://127.0.0.1:8000/docs


### ☁️ Deployment (optional)
Deploy using one of these free platforms:
- Render (recommended – easiest)
- Railway
- Google Cloud Run (free)
- HuggingFace Spaces (Docker allowed)

### 📁 Repository Structure
app/           -> FastAPI service  
models/        -> Saved model  
notebooks/     -> Training notebooks  
requirements.txt  
Dockerfile  
README.md


### 👩‍💻 Author
Marcia — Data Scientist | ML & CV
(ML Zoomcamp mid-project)