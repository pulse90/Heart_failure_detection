import os
import sys
import joblib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, recall_score, f1_score,
    RocCurveDisplay,
)
from xgboost import XGBClassifier
import shap

# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH   = "data/heart_failure_clinical_records.csv"
MODEL_PATH  = "model.pkl"
SCALER_PATH = "scaler.pkl"
RANDOM_STATE = 42

FEATURES = [
    "age", "anaemia", "creatinine_phosphokinase", "diabetes",
    "ejection_fraction", "high_blood_pressure", "platelets",
    "serum_creatinine", "serum_sodium", "sex", "smoking", "time",
]
TARGET = "DEATH_EVENT"


# ── 1. Load data ───────────────────────────────────────────────────────────────

def load_data():
    if not os.path.exists(DATA_PATH):
        print(f"\nERROR: Dataset not found at '{DATA_PATH}'")
        print("Download from Kaggle:")
        print("  https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data")
        print("Place the CSV at: data/heart_failure_clinical_records.csv\n")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)

    # Strip whitespace from column names (common CSV issue)
    df.columns = df.columns.str.strip()

    # Validate TARGET column exists
    if TARGET not in df.columns:
        print(f"\nERROR: Column '{TARGET}' not found in CSV.")
        print(f"Available columns: {df.columns.tolist()}\n")
        sys.exit(1)

    # Validate FEATURES columns exist
    missing_cols = [c for c in FEATURES if c not in df.columns]
    if missing_cols:
        print(f"\nERROR: Missing feature columns: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}\n")
        sys.exit(1)

    # Ensure TARGET is integer (handles float/string edge cases)
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)

    print(f"Loaded dataset : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Death rate     : {df[TARGET].mean():.1%}")
    print(f"Class balance  : {df[TARGET].value_counts().to_dict()}")

    # Warn if data looks suspicious (correct dataset has ~96 deaths)
    if df[TARGET].sum() < 10:
        print("\nWARNING: Very few positive (death) cases detected!")
        print(f"  Expected ~96 deaths (32%) — found only {df[TARGET].sum()}.")
        print("  Your CSV may be corrupted or wrong. Re-download from Kaggle.\n")

    return df


# ── 2. Preprocessing ──────────────────────────────────────────────────────────

def preprocess(df):
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    missing = X.isnull().sum().sum()
    if missing > 0:
        print(f"Warning: {missing} missing values found — filling with median")
        X = X.fillna(X.median())

    return X, y


# ── 3. EDA plots ──────────────────────────────────────────────────────────────

def plot_eda(df):
    os.makedirs("plots", exist_ok=True)

    # Only plot EDA if we have both classes
    if df[TARGET].nunique() < 2:
        print("Skipping EDA plots — only one class present in data.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Heart Failure — EDA: Key Clinical Features vs Death Event",
                 fontsize=14, fontweight="bold")

    key_features = [
        "ejection_fraction", "serum_creatinine", "age",
        "serum_sodium", "platelets", "creatinine_phosphokinase",
    ]

    for ax, feat in zip(axes.flatten(), key_features):
        for label, grp in df.groupby(TARGET)[feat]:
            grp.plot.kde(ax=ax, label=f"{'Died' if label == 1 else 'Survived'}")
        ax.set_title(feat.replace("_", " ").title())
        ax.set_xlabel("")
        ax.legend()

    plt.tight_layout()
    plt.savefig("plots/eda_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plots/eda_distributions.png")

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = df[FEATURES + [TARGET]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, linewidths=0.5)
    plt.title("Feature Correlation Matrix", fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: plots/correlation_heatmap.png")


# ── 4. Train models ───────────────────────────────────────────────────────────

def train_models(X_train, y_train):
    # Dynamic class weight for XGBoost
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    pos_weight = round(neg / pos, 2) if pos > 0 else 1
    print(f"\nClass counts — Survived: {neg} | Died: {pos} | XGB scale_pos_weight: {pos_weight}")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            verbosity=0,
        ),
    }

    # Use StratifiedKFold to preserve class ratio in CV folds
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("\nTraining models ...")
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=cv, scoring="roc_auc")
        print(f"  {name:<25} CV AUC-ROC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        trained[name] = model

    return trained


# ── 5. Evaluate ───────────────────────────────────────────────────────────────

def evaluate_models(trained_models, X_test, y_test):
    print("\n" + "=" * 60)
    print("Model Evaluation on Test Set")
    print("=" * 60)

    # Get labels actually present in test set (handles edge cases)
    present_labels = sorted(y_test.unique().tolist())
    label_map = {0: "Survived", 1: "Died"}
    target_names = [label_map[l] for l in present_labels]

    results = {}
    os.makedirs("plots", exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Confusion Matrices — All Models", fontsize=13, fontweight="bold")

    for i, (name, model) in enumerate(trained_models.items()):
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Handle case where only one class is present in test set
        if len(present_labels) < 2:
            print(f"\nWARNING: Only one class in test set — AUC-ROC cannot be computed.")
            auc = float("nan")
        else:
            auc = roc_auc_score(y_test, y_proba)

        recall = recall_score(y_test, y_pred, zero_division=0)
        f1     = f1_score(y_test, y_pred, zero_division=0)

        results[name] = {"AUC-ROC": auc, "Recall": recall, "F1": f1}

        print(f"\n{name}")
        print(f"  AUC-ROC : {auc:.4f}" if not np.isnan(auc) else "  AUC-ROC : N/A (single class in test set)")
        print(f"  Recall  : {recall:.4f}  <- most important for medical use")
        print(f"  F1 Score: {f1:.4f}")
        print(classification_report(
            y_test, y_pred,
            labels=present_labels,
            target_names=target_names,
            zero_division=0,
        ))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=present_labels)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=target_names,
                    yticklabels=target_names,
                    ax=axes[i])
        auc_str = f"{auc:.3f}" if not np.isnan(auc) else "N/A"
        axes[i].set_title(f"{name}\nAUC={auc_str} | Recall={recall:.3f}")
        axes[i].set_ylabel("Actual")
        axes[i].set_xlabel("Predicted")

    plt.tight_layout()
    plt.savefig("plots/confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved: plots/confusion_matrices.png")

    # ROC curves (only if both classes present)
    if len(present_labels) == 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, model in trained_models.items():
            RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
        ax.set_title("ROC Curves — All Models", fontweight="bold")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        plt.tight_layout()
        plt.savefig("plots/roc_curves.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: plots/roc_curves.png")

    # Summary table
    print("\n" + "=" * 60)
    print("Summary Comparison")
    print("=" * 60)
    results_df = pd.DataFrame(results).T
    print(results_df.round(4).to_string())

    return results


# ── 6. SHAP explainability ────────────────────────────────────────────────────

def plot_shap(model, X_test, model_name="XGBoost"):
    print(f"\nGenerating SHAP feature importance for {model_name} ...")
    os.makedirs("plots", exist_ok=True)

    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Random Forest returns list [class0, class1] — take class 1
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test,
                          feature_names=FEATURES,
                          plot_type="bar",
                          show=False)
        plt.title(f"SHAP Feature Importance — {model_name}", fontweight="bold")
        plt.tight_layout()
        plt.savefig("plots/shap_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: plots/shap_importance.png")

        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, X_test,
                          feature_names=FEATURES,
                          show=False)
        plt.title(f"SHAP Summary Plot — {model_name}", fontweight="bold")
        plt.tight_layout()
        plt.savefig("plots/shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved: plots/shap_summary.png")

    except Exception as e:
        print(f"Warning: SHAP plot failed — {e}")
        print("Skipping SHAP plots. Training still completed successfully.")


# ── 7. Save best model ────────────────────────────────────────────────────────

def save_best_model(trained_models, results, scaler):
    # Pick best by AUC-ROC, ignoring NaN values
    valid_results = {k: v for k, v in results.items() if not np.isnan(v["AUC-ROC"])}

    if not valid_results:
        # Fallback: pick by Recall if AUC is unavailable
        print("Warning: AUC-ROC unavailable — selecting best model by Recall.")
        best_name = max(results, key=lambda k: results[k]["Recall"])
    else:
        best_name = max(valid_results, key=lambda k: valid_results[k]["AUC-ROC"])

    best_model = trained_models[best_name]

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler,     SCALER_PATH)

    auc_val = results[best_name]["AUC-ROC"]
    auc_str = f"{auc_val:.4f}" if not np.isnan(auc_val) else "N/A"

    print(f"\nBest model : {best_name} (AUC-ROC: {auc_str})")
    print(f"Saved model  -> {MODEL_PATH}")
    print(f"Saved scaler -> {SCALER_PATH}")

    return best_name, best_model


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Heart Failure Detection — Training Pipeline")
    print("=" * 60)

    # 1. Load
    df = load_data()

    # 2. EDA plots
    plot_eda(df)

    # 3. Preprocess
    X, y = preprocess(df)

    # 4. Stratified split — preserves class ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\nTrain size : {len(X_train)} | Test size: {len(X_test)}")
    print(f"Train class balance: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Test  class balance: {dict(zip(*np.unique(y_test,  return_counts=True)))}")

    # 5. Scale features
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Keep as DataFrame so SHAP shows feature names
    X_train_sc = pd.DataFrame(X_train_sc, columns=FEATURES)
    X_test_sc  = pd.DataFrame(X_test_sc,  columns=FEATURES)

    # 6. Train
    trained_models = train_models(X_train_sc, y_train)

    # 7. Evaluate
    results = evaluate_models(trained_models, X_test_sc, y_test)

    # 8. Save best model + scaler
    best_name, best_model = save_best_model(trained_models, results, scaler)

    # 9. SHAP explainability (tree-based models only)
    if best_name in ("Random Forest", "XGBoost"):
        plot_shap(best_model, X_test_sc, best_name)
    else:
        plot_shap(trained_models["XGBoost"], X_test_sc, "XGBoost")

    print("\nTraining complete!")
    print("Next step: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()