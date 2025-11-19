#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from pathlib import Path
import joblib

# ------------------------
# Config
# ------------------------
MODEL_PATH = Path('models/xgboost_model.pkl')  # use the XGBoost model
DATA_PATH = Path('../data/diabetes_new_samples.csv')        # new samples to predict

# ------------------------
# Load model
# ------------------------
assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"
model = joblib.load(MODEL_PATH)
print(f"Loaded model from {MODEL_PATH}")

# ------------------------
# Load new data
# ------------------------
assert DATA_PATH.exists(), f"Data file not found: {DATA_PATH}"
df_new = pd.read_csv(DATA_PATH)

# X features (make sure it matches training columns)
X_new = df_new  # if df_new already has the same columns as train

# ------------------------
# Make predictions
# ------------------------
preds_proba = model.predict_proba(X_new)[:, 1]  # probability of class 1
preds = model.predict(X_new)                    # predicted class 0/1

df_new['predicted_class'] = preds
df_new['predicted_proba'] = preds_proba

print(df_new.head())

# Optionally save predictions
df_new.to_csv('predictions.csv', index=False)
print("Predictions saved to predictions.csv")
