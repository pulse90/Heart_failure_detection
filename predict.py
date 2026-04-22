import os
import joblib
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# ── Config ───────────────────────────────────────────────────

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

def _extract_shap_1d(shap_values, expected_value):
    """
    Normalise SHAP output to a 1-D array of length n_features (class-1 slice)
    and a scalar base value, regardless of what the explainer returned.

    Handles all known shapes:
        list of arrays  → pick index [1]
        (n_features,)   → already 1-D
        (1, n_features) → squeeze row 0
        (1, n_features, 2) → squeeze row 0, pick class 1   ← your case
        (n_features, 2)    → pick class 1
    """
    # ── resolve list output ──────────────────────────────────────────────────
    if isinstance(shap_values, list):
        # list[0] = class-0, list[1] = class-1
        sv = np.array(shap_values[1])
        ev = (expected_value[1]
              if isinstance(expected_value, (list, np.ndarray))
              else expected_value)
    else:
        sv = np.array(shap_values)
        ev = (expected_value[1]
              if isinstance(expected_value, (list, np.ndarray))
              else expected_value)

    # ── resolve ndim ─────────────────────────────────────────────────────────
    if sv.ndim == 1:
        # (n_features,)  — already done
        shap_1d = sv

    elif sv.ndim == 2:
        if sv.shape[0] == 1:
            # (1, n_features)
            shap_1d = sv[0]
        else:
            # (n_features, 2)  — last axis is classes
            shap_1d = sv[:, 1]

    elif sv.ndim == 3:
        # (n_samples, n_features, n_classes)  → row 0, class 1
        shap_1d = sv[0, :, 1]

    else:
        raise ValueError(f"Unexpected SHAP shape: {sv.shape}")

    return shap_1d, float(ev)


def _get_explainer_and_shap(model, input_scaled_df):
    """
    Returns:
        explainer, shap_vals_1d (length = n_features), base_value
    """
    model_name = type(model).__name__

    TREE_MODELS = {
        "RandomForestClassifier", "XGBClassifier",
        "GradientBoostingClassifier", "LGBMClassifier",
        "ExtraTreesClassifier", "DecisionTreeClassifier",
    }

    LINEAR_MODELS = {
        "LogisticRegression", "LinearSVC",
        "SGDClassifier", "RidgeClassifier",
    }

    # ── TREE MODELS ──────────────────────────────────────────────────────────
    if model_name in TREE_MODELS:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_scaled_df)
        shap_1d, base_value = _extract_shap_1d(shap_values, explainer.expected_value)

    # ── LINEAR MODELS ────────────────────────────────────────────────────────
    elif model_name in LINEAR_MODELS:
        background = pd.DataFrame(
            np.zeros((1, input_scaled_df.shape[1])),
            columns=input_scaled_df.columns,
        )
        explainer   = shap.LinearExplainer(model, background)
        shap_values = explainer.shap_values(input_scaled_df)
        shap_1d, base_value = _extract_shap_1d(shap_values, explainer.expected_value)

    # ── FALLBACK (KERNEL) ────────────────────────────────────────────────────
    else:
        background = pd.DataFrame(
            np.zeros((10, input_scaled_df.shape[1])),
            columns=input_scaled_df.columns,
        )
        explainer   = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(input_scaled_df)
        shap_1d, base_value = _extract_shap_1d(shap_values, explainer.expected_value)

    # ── Safety check ─────────────────────────────────────────────────────────
    if shap_1d.shape[0] != input_scaled_df.shape[1]:
        raise ValueError(
            f"SHAP feature mismatch: got {shap_1d.shape[0]}, "
            f"expected {input_scaled_df.shape[1]}"
        )

    return explainer, shap_1d, base_value


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

    plt.figure(figsize=(7, 4))
    shap.plots.waterfall(shap_explanation, show=False, max_display=12)

    ax = plt.gca()

    # Symmetric x-axis so positive and negative bars scale equally
    x_max = max(abs(shap_vals_1d.min()), abs(shap_vals_1d.max())) * 1.3
    ax.set_xlim(-x_max, x_max)

    # Light grid on the x-axis for readability
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    ax.set_title("SHAP Feature Contributions for This Patient",
                 fontsize=10, fontweight="bold", pad=8)
    ax.tick_params(axis="both", labelsize=7)

    plt.tight_layout()
    fig = plt.gcf()
    return fig


# ── Global Feature Importance ──────────────────────────────────────────────────

def get_feature_importance(model) -> pd.DataFrame:
    """
    Returns a DataFrame of global feature importances sorted descending.
    Works for tree-based and linear models.
    """
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_

        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])

        else:
            importances = np.ones(len(FEATURES)) / len(FEATURES)

        df = pd.DataFrame({
            "Feature":    [FEATURE_LABELS.get(f, f) for f in FEATURES],
            "Importance": importances,
        }).sort_values("Importance", ascending=False).reset_index(drop=True)

        return df

    except Exception:
        return pd.DataFrame({"Feature": FEATURES, "Importance": [0] * len(FEATURES)})