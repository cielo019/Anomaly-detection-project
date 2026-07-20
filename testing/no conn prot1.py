# -*- coding: utf-8 -*-
"""
stage2_behavioral_only.py

Stage 2 follow-up to shortcut_analysis_iot23.py.

The Stage 1 six-step methodology showed that port, protocol (proto), and
connection state (conn_state) are almost duplicate signals in this dataset
(Step 2), so removing port alone does NOT remove the shortcut - the model
just leans on proto/conn_state instead (Steps 1, 3, 4, 5).

This script runs the "next essential experiment" called out in the report's
Conclusion / Limitations section: remove port, proto, AND conn_state
TOGETHER, and test whether the remaining purely behavioural features
(duration, byte counts, packet counts, missed_bytes) can still separate
malicious from benign traffic on their own.

Two behavioural feature sets are tested:
  - BEHAVIORAL (IPs kept)   : drop id.orig_p, id.resp_p, proto, conn_state
                              (id.orig_h / id.resp_h kept, service kept)
  - BEHAVIORAL (strict)     : as above, but ALSO drop id.orig_h, id.resp_h
                              and service, leaving only duration / bytes /
                              packets / missed_bytes - i.e. nothing that
                              identifies "which connection/scenario" this is,
                              only "how the traffic behaved".

Six diagnostic steps, adapted from Stage 1 to this new comparison:

  STEP 1  Full vs Behavioral(IPs kept) vs Behavioral(strict) - does accuracy
          collapse when port+proto+conn_state are removed together?
  STEP 2  Residual correlation check - do the remaining behavioural columns
          still leak the port/proto/conn_state signal?
  STEP 3  Permutation importance on the behavioral-only models - what are
          they actually reading now?
  STEP 4  Test-time shuffle of the behavioral features - does the model
          collapse to chance, confirming it is using real signal (not some
          other leftover identifier)?
  STEP 5  Grouped train/test split (by source IP) on the behavioral-only
          model - does it generalize to connections from unseen hosts?
  STEP 6  Multi-seed repeat of Full vs Behavioral-only + paired t-test for
          statistical significance of the drop.

Expects netsec.csv (same tab-separated Zeek conn log) next to this file.
Each step prints its results and saves a CSV.
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

RANDOM_STATE = 42

# ==========================================
# LOAD + PREPROCESS (same as Stage 1)
# ==========================================

column_names = [
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
    "proto", "service", "duration", "orig_bytes", "resp_bytes",
    "conn_state", "local_orig", "local_resp", "missed_bytes", "history",
    "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
    "tunnel_parents", "Label", "extra_column"
]

script_dir = os.path.dirname(os.path.abspath(__file__))
candidate_paths = [
    os.path.join(script_dir, "netsec.csv"),
    os.path.join(script_dir, "Datasets", "netsec.csv"),
    os.path.join(script_dir, os.pardir, "Datasets", "netsec.csv"),
]

csv_path = None
for path in candidate_paths:
    if os.path.exists(path):
        csv_path = path
        break

if csv_path is None:
    raise FileNotFoundError(f"Could not find netsec.csv. Checked: {candidate_paths}")

df = pd.read_csv(
    csv_path, sep="\t", header=None, names=column_names,
    comment="#", engine="python"
)

if "extra_column" in df.columns and df["extra_column"].isnull().all():
    df.drop(columns=["extra_column"], inplace=True)

df = df.drop_duplicates().reset_index(drop=True)

selected_features = [
    "id.orig_p", "id.resp_p", "id.orig_h", "id.resp_h",
    "proto", "service", "duration", "orig_bytes", "resp_bytes",
    "conn_state", "missed_bytes", "orig_pkts", "orig_ip_bytes",
    "resp_pkts", "resp_ip_bytes", "Label"
]
df = df[selected_features]

df.replace(to_replace=r"^\s*-\s*$", value=np.nan, regex=True, inplace=True)

categorical_columns = ["proto", "service", "conn_state"]
df[categorical_columns] = df[categorical_columns].fillna("missing")

numeric_cols = [c for c in df.columns if c not in categorical_columns + ["Label"]]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

encoder = LabelEncoder()
for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col].astype(str))

ip_columns = ["id.orig_h", "id.resp_h"]
for col in ip_columns:
    df[col] = encoder.fit_transform(df[col].astype(str))

label_encoder = LabelEncoder()
df["Label"] = label_encoder.fit_transform(df["Label"])

df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

# ==========================================
# FEATURE SETS
# ==========================================

PORT_COLS = ["id.orig_p", "id.resp_p"]
SHORTCUT_COLS = ["id.orig_p", "id.resp_p", "proto", "conn_state"]   # the trio + port
IP_COLS = ["id.orig_h", "id.resp_h"]
BEHAVIORAL_COLS = [
    "duration", "orig_bytes", "resp_bytes",
    "missed_bytes", "orig_pkts", "orig_ip_bytes",
    "resp_pkts", "resp_ip_bytes"
]
GROUP_COL = "id.orig_h"   # generalization test: unseen source hosts

X_full = df.drop(columns=["Label"])
y = df["Label"]

# Behavioral, IPs + service kept (only port/proto/conn_state removed)
X_behav_ips = df.drop(columns=SHORTCUT_COLS + ["Label"])

# Strict behavioral: only duration/bytes/packets/missed_bytes - nothing
# that identifies "which connection/host/service" this is
X_behav_strict = df[BEHAVIORAL_COLS]

print("Dataset ready:", df.shape)
print("Full feature set        :", X_full.shape)
print("Behavioral (IPs kept)   :", X_behav_ips.shape, "->", list(X_behav_ips.columns))
print("Behavioral (strict)     :", X_behav_strict.shape, "->", list(X_behav_strict.columns))


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


def evaluate(X, y, label):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    rows = []
    for name, model in make_models().items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rows.append({"Model": name, "Feature Set": label, **score(y_test, y_pred)})
    return pd.DataFrame(rows)


# ==========================================
# STEP 1 - FULL vs BEHAVIORAL(IPs kept) vs BEHAVIORAL(strict)
# ==========================================

print("\n" + "=" * 60)
print("STEP 1: Full vs Behavioral(IPs kept) vs Behavioral(strict)")
print("=" * 60)

res_full = evaluate(X_full, y, "Full (with port/proto/conn_state)")
res_behav_ips = evaluate(X_behav_ips, y, "Behavioral (IPs kept)")
res_behav_strict = evaluate(X_behav_strict, y, "Behavioral (strict)")

step1_df = pd.concat([res_full, res_behav_ips, res_behav_strict], ignore_index=True).round(4)
print(step1_df.to_string(index=False))
step1_df.to_csv("Stage2_Step1_FeatureSet_Comparison.csv", index=False)

pivot = step1_df.pivot(index="Model", columns="Feature Set", values="F1 Score").round(4)
pivot["Drop (Full - IPs kept)"] = pivot["Full (with port/proto/conn_state)"] - pivot["Behavioral (IPs kept)"]
pivot["Drop (Full - strict)"] = pivot["Full (with port/proto/conn_state)"] - pivot["Behavioral (strict)"]
print("\nF1 comparison / drop:")
print(pivot.round(4).to_string())
pivot.to_csv("Stage2_Step1_F1_Drop_Summary.csv")
print("\nA large drop here (unlike Stage 1's ~0.002-0.007) would show that")
print("port/proto/conn_state together carried most of the signal, and the")
print("purely behavioral features cannot support the same accuracy alone.")


# ==========================================
# STEP 2 - RESIDUAL CORRELATION CHECK
# ==========================================

print("\n" + "=" * 60)
print("STEP 2: Do remaining features still leak port/proto/conn_state?")
print("=" * 60)

corr_matrix = X_full.corr(numeric_only=True)
residual_corr = corr_matrix[SHORTCUT_COLS].drop(index=SHORTCUT_COLS, errors="ignore")
print(residual_corr.round(3).to_string())
residual_corr.to_csv("Stage2_Step2_Residual_Correlations.csv")
print("\nHigh correlation between a remaining column and port/proto/conn_state")
print("would mean that column is itself a proxy shortcut and should be treated")
print("with the same suspicion as the trio removed above.")


# ==========================================
# STEP 3 - PERMUTATION IMPORTANCE ON BEHAVIORAL-ONLY MODELS
# ==========================================

print("\n" + "=" * 60)
print("STEP 3: Permutation importance - Behavioral (strict) models")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_behav_strict, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

for name, model in {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
    "LightGBM": LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1),
}.items():
    model.fit(X_train, y_train)
    perm = permutation_importance(
        model, X_test, y_test, n_repeats=10,
        random_state=RANDOM_STATE, scoring="f1_weighted"
    )
    imp_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance Mean": perm.importances_mean,
        "Importance Std": perm.importances_std,
    }).sort_values("Importance Mean", ascending=False)
    print(f"\n{name} - feature importance (behavioral-only model):")
    print(imp_df.to_string(index=False))
    imp_df.to_csv(f"Stage2_Step3_PermImportance_{name.replace(' ', '')}.csv", index=False)

print("\nIf importance now spreads across duration/bytes/packets rather than")
print("concentrating on one column, that is evidence of genuine behavioral")
print("learning rather than a single remaining shortcut field.")


# ==========================================
# STEP 4 - TEST-TIME SHUFFLE OF BEHAVIORAL FEATURES
# ==========================================

print("\n" + "=" * 60)
print("STEP 4: Shuffle ALL behavioral features at test time (strict set)")
print("=" * 60)

step4_results = []
for name, model in make_models().items():
    model.fit(X_train, y_train)

    y_pred_normal = model.predict(X_test)
    normal_scores = score(y_test, y_pred_normal)

    X_test_shuffled = X_test.copy()
    rng = np.random.RandomState(RANDOM_STATE)
    for col in BEHAVIORAL_COLS:
        X_test_shuffled[col] = rng.permutation(X_test_shuffled[col].values)

    y_pred_shuffled = model.predict(X_test_shuffled)
    shuffled_scores = score(y_test, y_pred_shuffled)

    step4_results.append({
        "Model": name,
        "F1 (Normal)": normal_scores["F1 Score"],
        "F1 (Behavioral Shuffled)": shuffled_scores["F1 Score"],
        "F1 Drop": normal_scores["F1 Score"] - shuffled_scores["F1 Score"],
    })

step4_df = pd.DataFrame(step4_results).round(4)
print(step4_df.to_string(index=False))
step4_df.to_csv("Stage2_Step4_BehavioralShuffle_Results.csv", index=False)
print("\nIf F1 collapses toward chance when these columns are scrambled, the")
print("model is genuinely relying on behavioral signal (not something else).")


# ==========================================
# STEP 5 - GROUPED SPLIT BY SOURCE IP (GENERALIZATION TEST)
# ==========================================

print("\n" + "=" * 60)
print(f"STEP 5: Grouped split by '{GROUP_COL}' - behavioral (strict) model")
print("=" * 60)

groups = df[GROUP_COL]
unique_groups = groups.dropna().unique()
if len(unique_groups) < 5:
    print("Skipping grouped split because there are too few unique source IP groups for a stable split.")
    step5_df = pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])
    step5_df.to_csv("Stage2_Step5_GroupedSplit_Results.csv", index=False)
else:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    try:
        train_idx, test_idx = next(gss.split(X_behav_strict, y, groups=groups))
    except ValueError as exc:
        print(f"Grouped split could not be created: {exc}")
        step5_df = pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])
        step5_df.to_csv("Stage2_Step5_GroupedSplit_Results.csv", index=False)
    else:
        X_train_g, X_test_g = X_behav_strict.iloc[train_idx], X_behav_strict.iloc[test_idx]
        y_train_g, y_test_g = y.iloc[train_idx], y.iloc[test_idx]

        overlap = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
        print(f"Source IPs shared between train and test: {len(overlap)} (should be 0)")
        print(f"Distinct source IPs total: {groups.nunique()}  <- check this isn't too small")

        step5_results = []
        for name, model in make_models().items():
            model.fit(X_train_g, y_train_g)
            y_pred = model.predict(X_test_g)
            step5_results.append({"Model": name, **score(y_test_g, y_pred)})

        step5_df = pd.DataFrame(step5_results).round(4)
        print("\nPerformance on connections from completely unseen source hosts:")
        print(step5_df.to_string(index=False))
        step5_df.to_csv("Stage2_Step5_GroupedSplit_Results.csv", index=False)


# ==========================================
# STEP 6 - MULTI-SEED SIGNIFICANCE TEST (FULL vs BEHAVIORAL STRICT)
# ==========================================

print("\n" + "=" * 60)
print("STEP 6: Multi-seed Full vs Behavioral(strict) + significance test")
print("=" * 60)

SEEDS = list(range(10))
seed_results = {name: {"full": [], "behav": []} for name in make_models().keys()}

for seed in SEEDS:
    Xf_train, Xf_test, y_train_s, y_test_s = train_test_split(
        X_full, y, test_size=0.20, random_state=seed, stratify=y
    )
    Xb_train, Xb_test, _, _ = train_test_split(
        X_behav_strict, y, test_size=0.20, random_state=seed, stratify=y
    )

    for name in make_models().keys():
        model_full = make_models()[name]
        model_full.fit(Xf_train, y_train_s)
        f1_full = f1_score(y_test_s, model_full.predict(Xf_test), average="weighted", zero_division=0)

        model_behav = make_models()[name]
        model_behav.fit(Xb_train, y_train_s)
        f1_behav = f1_score(y_test_s, model_behav.predict(Xb_test), average="weighted", zero_division=0)

        seed_results[name]["full"].append(f1_full)
        seed_results[name]["behav"].append(f1_behav)

print(f"\n{'Model':<22}{'Mean F1 Full':>14}{'Mean F1 Behav':>16}{'p-value (t-test)':>20}")
step6_rows = []
for name, s in seed_results.items():
    full_arr = np.array(s["full"])
    behav_arr = np.array(s["behav"])
    try:
        t_stat, p_val = ttest_rel(full_arr, behav_arr)
    except Exception:
        p_val = float("nan")
    print(f"{name:<22}{full_arr.mean():>14.4f}{behav_arr.mean():>16.4f}{p_val:>20.4g}")
    step6_rows.append({
        "Model": name,
        "Mean F1 (Full)": full_arr.mean(),
        "Mean F1 (Behavioral strict)": behav_arr.mean(),
        "Std F1 (Full)": full_arr.std(),
        "Std F1 (Behavioral strict)": behav_arr.std(),
        "Mean Drop": full_arr.mean() - behav_arr.mean(),
        "p-value": p_val,
    })

step6_df = pd.DataFrame(step6_rows).round(5)
step6_df.to_csv("Stage2_Step6_MultiSeed_Significance.csv", index=False)
print("\np < 0.05 means the drop from removing port+proto+conn_state together")
print("is statistically significant, not just noise from a single split.")

print("\n" + "=" * 60)
print("ALL STAGE 2 STEPS COMPLETE - see Stage2_Step1..Step6 CSV files")
print("=" * 60)