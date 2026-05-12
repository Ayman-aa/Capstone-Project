"""
train_models.py

Trains:
- Logistic Regression
- Random Forest
- XGBoost

Evaluates:
- AUC-ROC
- Accuracy
- False Positive Rate
- Profitability

Uses threshold optimization
with business constraints.
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay
)

from xgboost import XGBClassifier

# =========================================================
# BUSINESS SETTINGS
# =========================================================

INTEREST_PER_LOAN = 1500

LOSS_PER_DEFAULT = 12000

MAX_ALLOWED_FPR = 0.35

# =========================================================
# LOAD DATA
# =========================================================

print("Loading data...")

df = pd.read_csv("loan_data.csv")

df = pd.get_dummies(
    df,
    columns=[
        "HomeOwnership",
        "LoanPurpose",
        "DeviceType"
    ]
)

X = df.drop("LoanStatus", axis=1)

y = df["LoanStatus"]

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    stratify=y,

    random_state=42
)

# =========================================================
# SCALING
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

feature_names = X.columns.tolist()

# =========================================================
# CLASS BALANCE
# =========================================================

scale_pos_weight = (

    len(y_train[y_train == 0]) /

    len(y_train[y_train == 1])

)

# =========================================================
# MODELS
# =========================================================

models = {

    "Logistic Regression": {

        "model": LogisticRegression(

            max_iter=1000,

            random_state=42,

            class_weight="balanced"
        ),

        "scaled": True
    },

    "Random Forest": {

        "model": RandomForestClassifier(

            n_estimators=300,

            max_depth=12,

            random_state=42,

            n_jobs=-1,

            class_weight="balanced"
        ),

        "scaled": False
    },

    "XGBoost": {

        "model": XGBClassifier(

            n_estimators=300,

            max_depth=6,

            learning_rate=0.05,

            subsample=0.85,

            colsample_bytree=0.85,

            eval_metric="logloss",

            random_state=42,

            verbosity=0,

            scale_pos_weight=scale_pos_weight
        ),

        "scaled": False
    }
}

# =========================================================
# TRAINING
# =========================================================

results = {}

trained_models = {}

fig_roc, axes_roc = plt.subplots(
    1,
    3,
    figsize=(16, 5)
)

print("\n" + "=" * 60)

for i, (name, cfg) in enumerate(models.items()):

    model = cfg["model"]

    Xtr = X_train_scaled if cfg["scaled"] else X_train

    Xte = X_test_scaled if cfg["scaled"] else X_test

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    model.fit(Xtr, y_train)

    probs = model.predict_proba(Xte)[:, 1]

    # -----------------------------------------------------
    # THRESHOLD SEARCH
    # -----------------------------------------------------

    best_profit = -float("inf")

    best_threshold = 0.50

    # fallback prediction
    best_preds = (probs >= 0.50).astype(int)

    for threshold in np.arange(0.30, 0.80, 0.01):

        preds = (probs >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            preds
        ).ravel()

        profit = (
            tp * INTEREST_PER_LOAN
            - fp * LOSS_PER_DEFAULT
        )

        fpr = (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0
        )

        # keep only realistic lending strategies
        if (
            fpr < MAX_ALLOWED_FPR
            and profit > best_profit
        ):

            best_profit = profit

            best_threshold = threshold

            best_preds = preds

    # -----------------------------------------------------
    # FINAL PREDICTIONS
    # -----------------------------------------------------

    preds = best_preds

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        preds
    ).ravel()

    auc = roc_auc_score(y_test, probs)

    accuracy = (
        tp + tn
    ) / len(y_test)

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    # if no threshold satisfied conditions
    if best_profit == -float("inf"):

        best_profit = (
            tp * INTEREST_PER_LOAN
            - fp * LOSS_PER_DEFAULT
        )

    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------

    results[name] = {

        "AUC": round(float(auc), 4),

        "Accuracy": round(float(accuracy), 4),

        "FPR": round(float(fpr), 4),

        "Profit": int(best_profit),

        "Threshold": round(float(best_threshold), 2),

        "TP": int(tp),

        "FP": int(fp),

        "TN": int(tn),

        "FN": int(fn)
    }

    trained_models[name] = model

    # -----------------------------------------------------
    # PRINT RESULTS
    # -----------------------------------------------------

    print(f"\n{name}")

    print(f"  AUC       : {auc:.4f}")

    print(f"  Accuracy  : {accuracy:.2%}")

    print(f"  FPR       : {fpr:.2%}")

    print(f"  Threshold : {best_threshold:.2f}")

    print(f"  Profit    : ${best_profit:,.0f}")

    print(
        f"  CM        : "
        f"TP={tp}  FP={fp}  TN={tn}  FN={fn}"
    )

    # -----------------------------------------------------
    # ROC CURVE
    # -----------------------------------------------------

    RocCurveDisplay.from_predictions(

        y_test,

        probs,

        ax=axes_roc[i],

        name=name
    )

    axes_roc[i].plot(
        [0, 1],
        [0, 1],
        "--"
    )

    axes_roc[i].set_title(name)

print("=" * 60)

# =========================================================
# SAVE ROC CURVES
# =========================================================

plt.tight_layout()

plt.savefig(
    "roc_curves.png",
    dpi=130,
    bbox_inches="tight"
)

plt.close()

print("\nSaved roc_curves.png")

# =========================================================
# CONFUSION MATRICES
# =========================================================

fig_cm, axes_cm = plt.subplots(
    1,
    3,
    figsize=(14, 4)
)

for i, name in enumerate(results):

    r = results[name]

    cm = np.array([
        [r["TN"], r["FP"]],
        [r["FN"], r["TP"]]
    ])

    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues",

        cbar=False,

        ax=axes_cm[i],

        xticklabels=[
            "Pred Default",
            "Pred Paid"
        ],

        yticklabels=[
            "Actual Default",
            "Actual Paid"
        ]
    )

    axes_cm[i].set_title(name)

plt.tight_layout()

plt.savefig(
    "confusion_matrices.png",
    dpi=130,
    bbox_inches="tight"
)

plt.close()

print("Saved confusion_matrices.png")

# =========================================================
# PROFITABILITY CHART
# =========================================================

fig, ax = plt.subplots(figsize=(8, 4))

names = list(results.keys())

profits = [
    results[n]["Profit"]
    for n in names
]

bars = ax.bar(
    names,
    profits
)

ax.axhline(0)

ax.set_title("Profitability by Model")

ax.set_ylabel("Profit ($)")

for bar, value in zip(bars, profits):

    ax.text(

        bar.get_x() + bar.get_width() / 2,

        value,

        f"${value:,.0f}",

        ha="center",

        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    "profitability_chart.png",
    dpi=130,
    bbox_inches="tight"
)

plt.close()

print("Saved profitability_chart.png")

# =========================================================
# SAVE BEST MODEL
# =========================================================

best_model_name = max(
    results,
    key=lambda x: results[x]["Profit"]
)

print(
    f"\nBest model: {best_model_name}"
    f" (${results[best_model_name]['Profit']:,.0f})"
)

joblib.dump(
    trained_models[best_model_name],
    "best_model.pkl"
)

joblib.dump(
    trained_models,
    "all_models.pkl"
)

joblib.dump(
    scaler,
    "scaler.pkl"
)

joblib.dump(
    feature_names,
    "feature_names.pkl"
)

with open("model_results.json", "w") as f:

    json.dump(
        {
            "results": results,
            "best_model": best_model_name
        },
        f,
        indent=2
    )

print("\nSaved all model artifacts.")

print("\nRun next:")

print("streamlit run app.py")