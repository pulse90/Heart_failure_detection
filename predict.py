import os
import joblib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_PATH  = "model.pkl"
SCALER_PATH = "scaler.pkl"

FEATURES = [
    "age", "anaemia", "creatinine_phosphokinase", "diabetes",
    "ejection_fraction", "high_blood_pressure", "platelets",
    "serum_creatinine", "serum_sodium", "sex", "smoking", "time",
]

# Human-readable labels for display
FEATURE_LABELS = {
    "age":                      "Age (years)",
    "anaemia":                  "Anaemia",
    "creatinine_phosphokinase": "CPK Enzyme (mcg/L)",
    "diabetes":                 "Diabetes",
    "ejection_fraction":        "Ejection Fraction (%)",
    "high_blood_pressure":      "High Blood Pressure",
    "platelets":                "Platelets (kiloplatelets/mL)",
    "serum_creatinine":         "Serum Creatinine (mg/dL)",
    "serum_sodium":             "Serum Sodium (mEq/L)",
    "sex":                      "Sex",
    "smoking":                  "Smoking",
    "time":                     "Follow-up Period (days)",
}


# ── Load model ─────────────────────────────────────────────────────────────────

def load_model(model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    """
    Load saved model and scaler from disk.
    Raises FileNotFoundError with a helpful message if files are missing.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            "Run train.py first to generate model.pkl."
        )
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler file not found at '{scaler_path}'. "
            "Run train.py first to generate scaler.pkl."
        )

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


# ── Predict ────────────────────────────────────────────────────────────────────

def predict(patient_data: dict, model, scaler) -> dict:
    """
    Run prediction on a single patient.

    Args:
        patient_data : dict with keys matching FEATURES
        model        : trained sklearn/xgboost model
        scaler       : fitted StandardScaler

    Returns:
        dict with keys: prediction, probability, risk_level, input_df, input_scaled
    """
    # Build DataFrame in correct feature order
    input_df = pd.DataFrame([patient_data], columns=FEATURES)

    # Scale
    input_scaled = scaler.transform(input_df)
    input_scaled_df = pd.DataFrame(input_scaled, columns=FEATURES)

    # Predict
    prediction = int(model.predict(input_scaled)[0])
    probability = float(model.predict_proba(input_scaled)[0][1])

    # Risk level thresholds
    if probability >= 0.60:
        risk_level = "High"
    elif probability >= 0.35:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "prediction":   prediction,
        "probability":  probability,
        "risk_level":   risk_level,
        "input_df":     input_df,           # unscaled (for display)
        "input_scaled": input_scaled_df,    # scaled (for SHAP)
    }


# ── SHAP Explanation ───────────────────────────────────────────────────────────

def _get_explainer_and_shap(model, input_scaled_df):
    """
    Auto-selects the correct SHAP explainer based on model type.
    Returns (explainer, shap_values_1d) for the single input row.

    Supports:
      - TreeExplainer   : RandomForest, XGBoost, GradientBoosting, LightGBM
      - LinearExplainer : LogisticRegression, LinearSVC, SGDClassifier
      - KernelExplainer : any other model (slow fallback)
    """
    model_name = type(model).__name__

    TREE_MODELS   = {"RandomForestClassifier", "XGBClassifier",
                     "GradientBoostingClassifier", "LGBMClassifier",
                     "ExtraTreesClassifier", "DecisionTreeClassifier"}
    LINEAR_MODELS = {"LogisticRegression", "LinearSVC",
                     "SGDClassifier", "RidgeClassifier"}

    if model_name in TREE_MODELS:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_scaled_df)
        # RandomForest returns list [class0, class1] — take class 1
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        base_value = (
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        )

    elif model_name in LINEAR_MODELS:
        # LinearExplainer needs a background dataset — use zero vector as neutral baseline
        background  = pd.DataFrame(
            np.zeros((1, len(FEATURES))), columns=FEATURES
        )
        explainer   = shap.LinearExplainer(model, background)
        shap_values = explainer.shap_values(input_scaled_df)
        base_value  = (
            explainer.expected_value[0]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        )

    else:
        # Universal fallback — works for any model, slower
        background  = pd.DataFrame(
            np.zeros((10, len(FEATURES))), columns=FEATURES
        )
        explainer   = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(input_scaled_df)[1]
        base_value  = (
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        )

    # Flatten to 1D array (single patient row)
    shap_vals_1d = np.array(shap_values).flatten()

    return explainer, shap_vals_1d, float(base_value)


def explain(result: dict, model) -> plt.Figure:
    """
    Generate a SHAP waterfall chart for the predicted patient.

    Args:
        result : output dict from predict()
        model  : trained model

    Returns:
        matplotlib Figure
    """
    input_scaled_df = result["input_scaled"]
    input_df        = result["input_df"]

    explainer, shap_vals_1d, base_value = _get_explainer_and_shap(
        model, input_scaled_df
    )

    # Build SHAP Explanation object for waterfall plot
    shap_explanation = shap.Explanation(
        values        = shap_vals_1d,
        base_values   = base_value,
        data          = input_df.values[0],
        feature_names = [FEATURE_LABELS.get(f, f) for f in FEATURES],
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_explanation, show=False, max_display=12)
    plt.title("SHAP Feature Contributions for This Patient",
              fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()

    return fig


# ── Global Feature Importance ──────────────────────────────────────────────────

def get_feature_importance(model) -> pd.DataFrame:
    """
    Returns a DataFrame of global feature importances sorted descending.
    Works for tree-based and linear models.
    """
    model_name = type(model).__name__

    try:
        if hasattr(model, "feature_importances_"):
            # Tree-based: RandomForest, XGBoost etc.
            importances = model.feature_importances_

        elif hasattr(model, "coef_"):
            # Linear models: LogisticRegression etc.
            importances = np.abs(model.coef_[0])

        else:
            # Unknown model — return uniform importances
            importances = np.ones(len(FEATURES)) / len(FEATURES)

        df = pd.DataFrame({
            "Feature":    [FEATURE_LABELS.get(f, f) for f in FEATURES],
            "Importance": importances,
        }).sort_values("Importance", ascending=False).reset_index(drop=True)

        return df

    except Exception as e:
        # Return empty DataFrame on failure
        return pd.DataFrame({"Feature": FEATURES, "Importance": [0] * len(FEATURES)})
