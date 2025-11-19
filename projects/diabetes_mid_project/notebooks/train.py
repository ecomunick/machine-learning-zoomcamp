#!/usr/bin/env python
# coding: utf-8

# # Training Notebook
# Complete training pipeline for diabetes dataset.

# imports and helper functions
import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, accuracy_score, classification_report
import joblib

# plotting defaults
sns.set(style='whitegrid')


# small helpers
def plot_roc(model, X, y, title='ROC', savepath=None):
    probs = model.predict_proba(X)[:,1]
    fpr, tpr, _ = roc_curve(y, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'AUC={roc_auc:.3f}')
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(title)
    plt.legend()
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()

def plot_confusion(cm, labels=[0,1], savepath=None):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()

def ensure_dirs(outdir='models'):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    Path(outdir, 'plots').mkdir(parents=True, exist_ok=True)
    Path(outdir, 'artifacts').mkdir(parents=True, exist_ok=True)


# load data
DATA_PATH = Path('../data/diabetes_clean.csv')
assert DATA_PATH.exists(), f"File not found: {DATA_PATH}. Put your cleaned csv there."
df = pd.read_csv(DATA_PATH)

X = df.drop(columns='Outcome')
y = df['Outcome']

# ## Train / Validation / Test split (60/20/20)
RANDOM_STATE = 42
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=RANDOM_STATE)

print('Train/Val/Test shapes:', X_train.shape, X_val.shape, X_test.shape)

# ## Baseline: XGBoost
xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=RANDOM_STATE)
xgb.fit(X_train, y_train)
baseline_auc = roc_auc_score(y_val, xgb.predict_proba(X_val)[:,1])
print('Baseline XGBoost validation AUC:', baseline_auc)

# ## Hyperparameter tuning
xgb_param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1]
}

grid_xgb = GridSearchCV(
    XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=RANDOM_STATE),
    xgb_param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)
grid_xgb.fit(X_train, y_train)

best_xgb = grid_xgb.best_estimator_
print('Best XGB params:', grid_xgb.best_params_)

# Optional: cross-validation on full train+val
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(best_xgb, pd.concat([X_train, X_val]), pd.concat([y_train, y_val]), cv=skf, scoring='roc_auc')
print('Tuned XGB CV mean/std:', cv_scores.mean(), cv_scores.std())

# ## Retrain on full train+val
X_full = pd.concat([X_train, X_val]).reset_index(drop=True)
y_full = pd.concat([y_train, y_val]).reset_index(drop=True)
best_xgb.fit(X_full, y_full)

# ## Evaluate on Test
probs_test = best_xgb.predict_proba(X_test)[:,1]
preds_test = best_xgb.predict(X_test)

final_auc = roc_auc_score(y_test, probs_test)
acc = accuracy_score(y_test, preds_test)
cm = confusion_matrix(y_test, preds_test)
report = classification_report(y_test, preds_test, output_dict=True)

print('Final Test AUC:', final_auc)
print('Accuracy:', acc)
print('Confusion matrix:\n', cm)

# Plots
ensure_dirs('models')
plot_roc(best_xgb, X_test, y_test, title='ROC XGBoost', savepath='models/plots/roc_xgb.png')
plot_confusion(cm, savepath='models/plots/cm_xgb.png')

# ## Save model & metadata
model_path = Path('models') / 'xgboost_model.pkl'
joblib.dump(best_xgb, model_path)

metadata = {
    'selected_model': 'XGBoost',
    'best_params': grid_xgb.best_params_,
    'final_test_auc': float(final_auc),
    'final_test_accuracy': float(acc),
    'confusion_matrix': cm.tolist(),
    'classification_report': report,
    'timestamp': datetime.utcnow().isoformat()
}

(Path('models') / 'artifacts' / 'metadata.json').write_text(json.dumps(metadata, indent=2))
print('Saved model and metadata to models/')