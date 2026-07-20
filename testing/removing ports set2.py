# -*- coding: utf-8 -*-
"""
DDoS Flow-Based Dataset - Model Comparison Script
Adapted from the original Zeek/conn-log pipeline (test.py) to work with the
CICIDS2017-style CSV: Friday-WorkingHours-Afternoon-DDos_pcap_ISCX.csv

This file is comma-separated with a header row and ~79 flow-statistics
columns (CICFlowMeter output), not the tab-separated Zeek conn log the
original script was written for - so the loading/cleaning section is
rewritten, but the modeling / evaluation section keeps the same structure.
"""

import pandas as pd
import numpy as np
import os

# Provide a notebook-style `display` when running as a script
try:
    from IPython.display import display
except Exception:
    def display(x):
        try:
            print(x.head())
        except Exception:
            print(x)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ==========================================
# LOAD DATASET
# ==========================================

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "F:\\anomaly detection\\testing\\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

df = pd.read_csv(csv_path)

# Column names in this file have stray leading/trailing spaces (e.g.
# " Destination Port") - clean them up.
df.columns = [c.strip() for c in df.columns]

# ==========================================
# DATASET INFORMATION
# ==========================================

print(df.shape)
display(df.head())
print(df.columns.tolist())

# ==========================================
# CHECK DUPLICATES
# ==========================================

duplicates = df.duplicated().sum()
print("Duplicate Rows :", duplicates)
print("Percentage :", duplicates / len(df) * 100)

df = df.drop_duplicates().reset_index(drop=True)
print("Dataset after removing duplicates:", df.shape)

# ==========================================
# CLEAN NUMERIC DATA
# ==========================================

# All columns except Label are numeric flow statistics in this dataset.
# "Flow Bytes/s" and "Flow Packets/s" can contain +/-inf when duration is 0.
feature_cols = [c for c in df.columns if c != "Label"]

for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

# ==========================================
# ENCODE LABEL
# ==========================================

label_encoder = LabelEncoder()
df["Label"] = label_encoder.fit_transform(df["Label"])
print("Label classes:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# ==========================================
# FEATURE CORRELATION
# ==========================================

correlation = df.corr(numeric_only=True)["Label"]
print(correlation.sort_values(ascending=False))

# ==========================================
# CREATE DATASETS
# ==========================================

# All features (includes Destination Port)
X_full = df.drop(columns=["Label"])

# Remove the port feature (analogous to dropping IP/Port in the original script)
X_no_port = df.drop(columns=["Destination Port", "Label"])

y = df["Label"]

print("FULL DATASET :", X_full.shape)
print("WITHOUT PORT :", X_no_port.shape)
print("Labels :", y.shape)

# ==========================================
# MACHINE LEARNING MODELS
# ==========================================

models = {

    "Logistic Regression":
    Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Decision Tree":
    DecisionTreeClassifier(random_state=42),

    "Random Forest":
    RandomForestClassifier(n_estimators=100, random_state=42),

    "SVM":
    Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC())
    ]),

    "KNN":
    Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier())
    ]),

    "XGBoost":
    XGBClassifier(random_state=42, eval_metric="logloss"),

    "LightGBM":
    LGBMClassifier(random_state=42, verbosity=-1)

}

# ==========================================
# MODEL EVALUATION FUNCTION
# ==========================================

def evaluate_models(X, y):

    X = X.replace([np.inf, -np.inf], np.nan)
    combined = pd.concat([X, y], axis=1)

    print("\nMissing values in each column:")
    print(combined.isna().sum())

    combined = combined.dropna()

    X = combined.drop(columns=["Label"])
    y = combined["Label"]

    print("Rows before split:", len(X))
    print("Labels before split:", len(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("=" * 50)
    print("Training Samples :", len(X_train))
    print("Validation Samples :", len(X_test))
    print("=" * 50)

    results = []

    for name, model in models.items():

        print("Training :", name)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0)
        })

    return pd.DataFrame(results)

# ==========================================
# TRAIN WITH PORT FEATURE
# ==========================================

results_full = evaluate_models(X_full, y)
print(results_full)

# ==========================================
# TRAIN WITHOUT PORT FEATURE
# ==========================================

results_no_port = evaluate_models(X_no_port, y)
print(results_no_port)

# ==========================================
# COMPARE RESULTS
# ==========================================

comparison = results_full.merge(
    results_no_port,
    on="Model",
    suffixes=(" (Full)", " (No Port)")
)

comparison = comparison.round(4)
display(comparison)

# ==========================================
# PERFORMANCE DROP
# ==========================================

comparison["Accuracy Drop"] = comparison["Accuracy (Full)"] - comparison["Accuracy (No Port)"]
comparison["Precision Drop"] = comparison["Precision (Full)"] - comparison["Precision (No Port)"]
comparison["Recall Drop"] = comparison["Recall (Full)"] - comparison["Recall (No Port)"]
comparison["F1 Drop"] = comparison["F1 Score (Full)"] - comparison["F1 Score (No Port)"]

comparison = comparison.round(4)

print("\n========== FINAL COMPARISON ==========\n")
display(comparison)

# ==========================================
# SAVE RESULTS
# ==========================================

comparison.to_csv("Model_Comparison_DDoS.csv", index=False)
print("Results saved as Model_Comparison_DDoS.csv")

# ==========================================
# 5-FOLD CROSS VALIDATION (on full feature set)
# ==========================================

print("\n========== CROSS VALIDATION ==========\n")

combined_cv = pd.concat([X_full, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
X_cv = combined_cv.drop(columns=["Label"])
y_cv = combined_cv["Label"]

cv_results = []

for name, model in models.items():

    print("Running :", name)

    scores = cross_val_score(model, X_cv, y_cv, cv=5, scoring="f1_weighted")

    print("Scores :", scores)
    print("Average :", scores.mean())
    print()

    cv_results.append({
        "Model": name,
        "CV Mean F1": scores.mean(),
        "CV Std": scores.std()
    })

cv_results = pd.DataFrame(cv_results)
display(cv_results)

cv_results.to_csv("CrossValidationResults_DDoS.csv", index=False)
print("Cross-validation results saved.")

# ==========================================
# FINAL SUMMARY
# ==========================================

print("=" * 60)
print("SUMMARY")
print("=" * 60)

print("\nDataset Size :", len(df))
print("Number of Features (Full) :", X_full.shape[1])
print("Number of Features (No Port) :", X_no_port.shape[1])
print("\nDuplicate Rows :", duplicates)

print("\nModel Comparison")
display(comparison)

print("\nCross Validation")
display(cv_results)

print("\nExperiment Completed Successfully")