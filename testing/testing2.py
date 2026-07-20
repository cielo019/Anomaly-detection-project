# -*- coding: utf-8 -*-
"""
shortcut_analysis_cicids2017.py

Six-step methodology to test whether IDS models trained on the
CICIDS2017 DDoS flow-based dataset genuinely learn malicious traffic
behavior, or rely on the Destination Port feature as a shortcut.

Builds on the original test2.py pipeline (same loading, cleaning, and
encoding steps) and adds:

  STEP 1  Single-feature model trained on port only
  STEP 2  Multicollinearity check: port vs other kept features
  STEP 3  Permutation importance (+ SHAP if available) on full model
  STEP 4  Test-time port-shuffle robustness test
  STEP 5  Grouped train/test split by port (real generalization test)
  STEP 6  Multi-seed repeat of full-vs-no-port comparison + significance

Expects the CICIDS2017 CSV (Friday-WorkingHours-Afternoon-DDos...) next
to this file - update csv_path below if yours lives elsewhere. Each step
prints its results and also saves a CSV so you can drop the numbers
straight into a results table.
"""

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from scipy.stats import ttest_rel

warnings.filterwarnings("ignore")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

RANDOM_STATE = 42

# ==========================================
# LOAD + PREPROCESS (same as original test2.py)
# ==========================================

script_dir = os.path.dirname(os.path.abspath(__file__))
# NOTE: update this path to wherever your CICIDS2017 CSV actually lives
csv_path = os.path.join(script_dir, "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

df = pd.read_csv(csv_path)
df.columns = [c.strip() for c in df.columns]

df = df.drop_duplicates().reset_index(drop=True)

feature_cols = [c for c in df.columns if c != "Label"]
for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

label_encoder = LabelEncoder()
df["Label"] = label_encoder.fit_transform(df["Label"])

df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

PORT_COLS = ["Destination Port"]
GROUP_COL = "Destination Port"   # entity used for the grouped split in Step 5

X_full = df.drop(columns=["Label"])
y = df["Label"]

print("Dataset ready:", X_full.shape, " Labels:", y.shape)
print("Unique Destination Port values:", df[GROUP_COL].nunique())


# ==========================================
# MODEL ZOO + HELPERS
# ==========================================

def make_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
        "LightGBM": LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1),
    }


def score(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


# ==========================================
# STEP 1 - SINGLE-FEATURE MODEL (PORT ONLY)
# ==========================================

print("\n" + "=" * 60)
print("STEP 1: Single-feature model trained on Destination Port ONLY")
print("=" * 60)

X_port_only = df[PORT_COLS]

X_train, X_test, y_train, y_test = train_test_split(
    X_port_only, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

step1_results = []
for name, model in {
    "Decision Tree (depth=3)": DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE),
    "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]),
}.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    result = {"Model": name, **score(y_test, y_pred)}
    step1_results.append(result)
    print(name, "->", result)

step1_df = pd.DataFrame(step1_results)
step1_df.to_csv("Step1_PortOnly_Results_CICIDS.csv", index=False)
print("\nIf F1 here is high, Destination Port alone is a strong shortcut for the label.")


# ==========================================
# STEP 2 - MULTICOLLINEARITY CHECK
# ==========================================

print("\n" + "=" * 60)
print("STEP 2: Correlation between port and other kept features")
print("=" * 60)

corr_matrix = X_full.corr(numeric_only=True)
port_corr = corr_matrix[PORT_COLS].drop(index=PORT_COLS, errors="ignore")
print(port_corr.sort_values(by=PORT_COLS[0], ascending=False).head(15))
port_corr.to_csv("Step2_Port_Feature_Correlations_CICIDS.csv")
print("\nHigh correlation between Destination Port and another kept feature means")
print("the shortcut signal survives even after dropping port, via that feature.")


# ==========================================
# STEP 3 - PERMUTATION IMPORTANCE (+ SHAP)
# ==========================================

print("\n" + "=" * 60)
print("STEP 3: Permutation importance on full-feature models")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

for name, model in {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
    "LightGBM": LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1),
}.items():
    model.fit(X_train, y_train)
    perm = permutation_importance(
        model, X_test, y_test, n_repeats=5,   # fewer repeats: this dataset has 77 features and 44k+ test rows
        random_state=RANDOM_STATE, scoring="f1_weighted"
    )
    imp_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance Mean": perm.importances_mean,
        "Importance Std": perm.importances_std,
    }).sort_values("Importance Mean", ascending=False)
    print(f"\n{name} - top 10 features by permutation importance:")
    print(imp_df.head(10).to_string(index=False))
    imp_df.to_csv(f"Step3_PermImportance_{name.replace(' ', '')}_CICIDS.csv", index=False)

if HAS_SHAP:
    print("\nComputing SHAP values for Random Forest on a sample (this can take a while)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    explainer = shap.TreeExplainer(rf)
    sample = X_test.sample(min(500, len(X_test)), random_state=RANDOM_STATE)
    shap_values = explainer.shap_values(sample)
    print("SHAP values computed. Run shap.summary_plot(shap_values, sample)")
    print("interactively (e.g. in a notebook) to visualize feature impact.")
else:
    print("\n(shap not installed - skipping SHAP step. `pip install shap` to enable it.)")


# ==========================================
# STEP 4 - TEST-TIME PORT-SHUFFLE ROBUSTNESS TEST
# ==========================================

print("\n" + "=" * 60)
print("STEP 4: Shuffle Destination Port at test time, re-score")
print("=" * 60)

step4_results = []
for name, model in make_models().items():
    model.fit(X_train, y_train)

    y_pred_normal = model.predict(X_test)
    normal_scores = score(y_test, y_pred_normal)

    X_test_shuffled = X_test.copy()
    rng = np.random.RandomState(RANDOM_STATE)
    for col in PORT_COLS:
        X_test_shuffled[col] = rng.permutation(X_test_shuffled[col].values)

    y_pred_shuffled = model.predict(X_test_shuffled)
    shuffled_scores = score(y_test, y_pred_shuffled)

    step4_results.append({
        "Model": name,
        "F1 (Normal)": normal_scores["F1 Score"],
        "F1 (Port Shuffled)": shuffled_scores["F1 Score"],
        "F1 Drop": normal_scores["F1 Score"] - shuffled_scores["F1 Score"],
    })

step4_df = pd.DataFrame(step4_results).round(4)
print(step4_df)
step4_df.to_csv("Step4_PortShuffle_Results_CICIDS.csv", index=False)
print("\nA large F1 Drop here means the model actively relies on Destination Port")
print("at inference time, not just happens to be correlated with it.")


# ==========================================
# STEP 5 - GROUPED TRAIN/TEST SPLIT (REAL GENERALIZATION TEST)
# ==========================================

print("\n" + "=" * 60)
print(f"STEP 5: Grouped split by '{GROUP_COL}' (ports unseen at train time)")
print("=" * 60)

groups = df[GROUP_COL]
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X_full, y, groups=groups))

X_train_g, X_test_g = X_full.iloc[train_idx], X_full.iloc[test_idx]
y_train_g, y_test_g = y.iloc[train_idx], y.iloc[test_idx]

overlap = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
print(f"Port values shared between train and test: {len(overlap)} (should be 0)")

step5_results = []
for name, model in make_models().items():
    model.fit(X_train_g, y_train_g)
    y_pred = model.predict(X_test_g)
    step5_results.append({"Model": name, **score(y_test_g, y_pred)})

step5_df = pd.DataFrame(step5_results).round(4)
print("\nPerformance on completely unseen Destination Ports:")
print(step5_df)
step5_df.to_csv("Step5_GroupedSplit_Results_CICIDS.csv", index=False)
print("\nCompare this against the original random-split results (~0.998-1.0 F1).")
print("A large drop here is the clearest evidence of port-based shortcut learning.")
print("(Note: if this dataset only has a handful of unique Destination Port values")
print("for the attack class, the grouped split may be small/imbalanced - check")
print("df[GROUP_COL].value_counts() to sanity-check group sizes before trusting this.)")


# ==========================================
# STEP 6 - MULTI-SEED REPEAT + SIGNIFICANCE TEST
# ==========================================

print("\n" + "=" * 60)
print("STEP 6: Multi-seed full-vs-no-port comparison + significance test")
print("=" * 60)

X_no_port = df.drop(columns=PORT_COLS + ["Label"])
SEEDS = list(range(10))

seed_results = {name: {"full": [], "no_port": []} for name in make_models().keys()}

for seed in SEEDS:
    Xf_train, Xf_test, y_train_s, y_test_s = train_test_split(
        X_full, y, test_size=0.20, random_state=seed, stratify=y
    )
    Xn_train, Xn_test, _, _ = train_test_split(
        X_no_port, y, test_size=0.20, random_state=seed, stratify=y
    )

    for name in make_models().keys():
        model_full = make_models()[name]
        model_full.fit(Xf_train, y_train_s)
        f1_full = f1_score(y_test_s, model_full.predict(Xf_test), average="weighted", zero_division=0)

        model_np = make_models()[name]
        model_np.fit(Xn_train, y_train_s)
        f1_no_port = f1_score(y_test_s, model_np.predict(Xn_test), average="weighted", zero_division=0)

        seed_results[name]["full"].append(f1_full)
        seed_results[name]["no_port"].append(f1_no_port)

print(f"\n{'Model':<22}{'Mean F1 Full':>14}{'Mean F1 NoPort':>16}{'p-value (t-test)':>20}")
step6_rows = []
for name, s in seed_results.items():
    full_arr = np.array(s["full"])
    no_port_arr = np.array(s["no_port"])
    try:
        t_stat, p_val = ttest_rel(full_arr, no_port_arr)
    except Exception:
        p_val = float("nan")
    print(f"{name:<22}{full_arr.mean():>14.4f}{no_port_arr.mean():>16.4f}{p_val:>20.4g}")
    step6_rows.append({
        "Model": name,
        "Mean F1 (Full)": full_arr.mean(),
        "Mean F1 (No Port)": no_port_arr.mean(),
        "Std F1 (Full)": full_arr.std(),
        "Std F1 (No Port)": no_port_arr.std(),
        "p-value": p_val,
    })

step6_df = pd.DataFrame(step6_rows).round(5)
step6_df.to_csv("Step6_MultiSeed_Significance_CICIDS.csv", index=False)
print("\np < 0.05 means the drop from removing port is statistically significant,")
print("not just noise from a single split.")

print("\n" + "=" * 60)
print("ALL STEPS COMPLETE - see Step1..Step6 *_CICIDS.csv files for full results")
print("=" * 60)