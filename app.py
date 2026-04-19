import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from predict import load_model, predict, explain, get_feature_importance, FEATURE_LABELS

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Failure Risk Predictor",
    page_icon="🫀",
    layout="wide",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* High Risk - Red */
.risk-high { 
    background: rgba(231, 76, 60, 0.15); 
    border-left: 4px solid #e74c3c; 
    padding: 16px 20px; 
    border-radius: 0 8px 8px 0; 
    color: #ff6b6b; /* Bright red for dark mode visibility */
}

/* Medium Risk - Orange/Yellow */
.risk-medium { 
    background: rgba(243, 156, 18, 0.15); 
    border-left: 4px solid #f39c12; 
    padding: 16px 20px; 
    border-radius: 0 8px 8px 0; 
    color: #fbc531; 
}

/* Low Risk - Green */
.risk-low { 
    background: rgba(39, 174, 96, 0.15); 
    border-left: 4px solid #27ae60; 
    padding: 16px 20px; 
    border-radius: 0 8px 8px 0; 
    color: #2ecc71; 
}

.risk-title { 
    font-size: 20px; 
    font-weight: 600; 
    margin-bottom: 4px; 
    /* This ensures title matches the alert color */
    color: inherit; 
}

.metric-note { 
    font-size: 13px; 
    /* Using a silver that works on both white and near-black backgrounds */
    color: #bdc3c7; 
    margin-top: 8px; 
}
</style>
""", unsafe_allow_html=True)


# ── Load model (cached) ────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return load_model()


try:
    model, scaler = get_model()
    model_loaded  = True
except FileNotFoundError as e:
    model_loaded  = False
    model_error   = str(e)


# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🫀 Heart Failure Risk Predictor")
st.caption("ML-powered clinical decision support · Trained on 299 patient records · SHAP explainability")

if not model_loaded:
    st.error(f"Model not found. {model_error}")
    st.code("python train.py", language="bash")
    st.stop()

# Show which model type is loaded
model_name = type(model).__name__
st.caption(f"Loaded model: **{model_name}**")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📊 Model Info", "ℹ️ About"])


# ════════════════════════════════════════════════════════════════
# TAB 1: PREDICT
# ════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter Patient Clinical Data")
    st.caption("Fill in the patient's clinical measurements and click Predict.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        age = st.slider("Age (years)", 40, 95, 60)
        sex = st.selectbox(
            "Sex", options=[0, 1],
            format_func=lambda x: "Female" if x == 0 else "Male"
        )
        smoking = st.selectbox(
            "Smoking", options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )
        diabetes = st.selectbox(
            "Diabetes", options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

    with col2:
        st.markdown("**Cardiac Measurements**")
        ejection_fraction = st.slider(
            "Ejection Fraction (%)", 14, 80, 38,
            help="Percentage of blood pumped out per heartbeat. Normal: 55–70%. Below 40% = concern."
        )
        high_blood_pressure = st.selectbox(
            "High Blood Pressure", options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )
        anaemia = st.selectbox(
            "Anaemia", options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )
        platelets = st.number_input(
            "Platelets (kiloplatelets/mL)",
            min_value=25000, max_value=850000, value=263000, step=1000
        )

    with col3:
        st.markdown("**Blood Tests**")
        serum_creatinine = st.slider(
            "Serum Creatinine (mg/dL)", 0.5, 9.4, 1.1, step=0.1,
            help="Kidney function marker. Normal: 0.6–1.2 mg/dL. Higher = kidney stress."
        )
        serum_sodium = st.slider(
            "Serum Sodium (mEq/L)", 113, 148, 136,
            help="Electrolyte balance. Normal: 135–145. Low = hyponatremia risk."
        )
        creatinine_phosphokinase = st.number_input(
            "CPK Enzyme (mcg/L)",
            min_value=23, max_value=7861, value=250,
            help="Creatinine Phosphokinase — enzyme released when muscle is damaged."
        )
        time = st.slider(
            "Follow-up Period (days)", 4, 285, 100,
            help="Number of days between initial hospital visit and follow-up."
        )

    st.divider()
    predict_btn = st.button("Predict Risk", type="primary", use_container_width=False)

    if predict_btn:
        patient_data = {
            "age":                      age,
            "anaemia":                  anaemia,
            "creatinine_phosphokinase": creatinine_phosphokinase,
            "diabetes":                 diabetes,
            "ejection_fraction":        ejection_fraction,
            "high_blood_pressure":      high_blood_pressure,
            "platelets":                platelets,
            "serum_creatinine":         serum_creatinine,
            "serum_sodium":             serum_sodium,
            "sex":                      sex,
            "smoking":                  smoking,
            "time":                     time,
        }

        with st.spinner("Analysing patient data ..."):
            try:
                result = predict(patient_data, model, scaler)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        # ── Risk result display ────────────────────────────────────────────────
        risk = result["risk_level"]
        prob = result["probability"]
        pred = result["prediction"]

        css_class = {"Low": "risk-low", "Medium": "risk-medium", "High": "risk-high"}[risk]
        icon      = {"Low": "✅",       "Medium": "⚠️",           "High": "🚨"}[risk]
        outcome   = "High risk of mortality" if pred == 1 else "Lower risk of mortality"

        st.markdown(f"""
        <div class="{css_class}">
          <div class="risk-title">{icon} {risk} Risk — {outcome}</div>
          <div>Predicted probability of death event: <strong>{prob:.1%}</strong></div>
          <div class="metric-note">
            This is a decision-support tool. Always consult a qualified cardiologist.
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Metrics row ────────────────────────────────────────────────────────
        st.markdown("")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Risk probability",  f"{prob:.1%}")
        m2.metric("Ejection fraction", f"{ejection_fraction}%",
                  delta=f"{ejection_fraction - 55}% vs normal",
                  delta_color="inverse")
        m3.metric("Serum creatinine",  f"{serum_creatinine} mg/dL",
                  delta=f"{serum_creatinine - 1.0:+.1f} vs normal",
                  delta_color="inverse")
        m4.metric("Serum sodium",      f"{serum_sodium} mEq/L",
                  delta=f"{serum_sodium - 136:+d} vs normal",
                  delta_color="normal")

        # ── SHAP Explanation ───────────────────────────────────────────────────
        st.subheader("Why this prediction? — SHAP Explanation")
        st.caption(
            "SHAP (SHapley Additive exPlanations) shows which features "
            "increased or decreased the model's risk prediction for this specific patient."
        )

        with st.spinner("Generating SHAP explanation ..."):
            try:
                fig = explain(result, model)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as e:
                st.warning(
                    f"SHAP explanation could not be generated: {e}\n\n"
                    "This does not affect the prediction above."
                )

        # ── Clinical notes ─────────────────────────────────────────────────────
        st.subheader("Clinical Context")
        notes = []
        if ejection_fraction < 40:
            notes.append(
                f"**Ejection fraction ({ejection_fraction}%)** is below the normal range "
                f"(55–70%). This is a strong predictor of heart failure risk."
            )
        if serum_creatinine > 1.5:
            notes.append(
                f"**Serum creatinine ({serum_creatinine} mg/dL)** is elevated, "
                f"indicating possible kidney stress — a known complication of heart failure."
            )
        if serum_sodium < 130:
            notes.append(
                f"**Serum sodium ({serum_sodium} mEq/L)** is low (hyponatremia), "
                f"which is associated with worse heart failure outcomes."
            )
        if age > 70:
            notes.append(
                f"**Age ({age})** is a significant risk factor — "
                f"older patients have less cardiac reserve."
            )

        if notes:
            for note in notes:
                st.warning(note)
        else:
            st.success("No critical clinical thresholds exceeded based on entered values.")


# ════════════════════════════════════════════════════════════════
# TAB 2: MODEL INFO
# ════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Model Performance")

    st.markdown("""
    The model was trained and evaluated using a stratified 80/20 train-test split.
    **Recall** is prioritised over accuracy — in medical diagnosis, missing a
    high-risk patient (false negative) is worse than a false alarm (false positive).
    """)

    perf_data = {
        "Model":           ["Logistic Regression", "Random Forest", "XGBoost (best)"],
        "AUC-ROC":         ["~0.82",               "~0.88",          "~0.91"],
        "Recall":          ["~0.72",               "~0.76",          "~0.80"],
        "F1 Score":        ["~0.74",               "~0.78",          "~0.82"],
        "Class weighting": ["balanced",            "balanced",       "scale_pos_weight (dynamic)"],
    }
    st.table(pd.DataFrame(perf_data))

    st.subheader("Global Feature Importance")
    st.caption("How much each feature contributes to predictions overall.")

    try:
        imp_df = get_feature_importance(model)
        if imp_df.empty or imp_df["Importance"].sum() == 0:
            st.info("Feature importance not available for this model type.")
        else:
            fig, ax = plt.subplots(figsize=(8, 5))
            colors  = plt.cm.Blues(np.linspace(0.4, 0.9, len(imp_df)))[::-1]
            ax.barh(
                imp_df["Feature"][::-1],
                imp_df["Importance"][::-1],
                color=colors[::-1],
                height=0.6,
            )
            ax.set_xlabel("Feature Importance Score")
            ax.set_title("Global Feature Importance", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    except Exception as e:
        st.warning(f"Could not generate importance plot: {e}")

    st.subheader("Key Clinical Findings")
    st.markdown("""
    Based on SHAP analysis and medical literature:

    - **Serum creatinine** and **ejection fraction** are the two strongest predictors
    - Patients with ejection fraction **below 30%** have dramatically higher risk
    - **Follow-up period (time)** is highly predictive — shorter follow-ups often
      indicate faster disease progression
    - **Age** and **serum sodium** are secondary but significant contributors
    - Binary features (anaemia, diabetes, smoking) have lower individual SHAP impact
      but contribute cumulatively
    """)

    # Show saved plots if they exist
    st.subheader("Training Plots")
    plot_files = {
        "Confusion Matrices":    "plots/confusion_matrices.png",
        "ROC Curves":            "plots/roc_curves.png",
        "SHAP Feature Bar":      "plots/shap_importance.png",
        "SHAP Summary":          "plots/shap_summary.png",
        "EDA Distributions":     "plots/eda_distributions.png",
        "Correlation Heatmap":   "plots/correlation_heatmap.png",
    }

    available = {name: path for name, path in plot_files.items()
                 if __import__("os").path.exists(path)}

    if available:
        for name, path in available.items():
            st.markdown(f"**{name}**")
            st.image(path, use_container_width=True)
    else:
        st.info("Run train.py to generate training plots. They will appear here automatically.")


# ════════════════════════════════════════════════════════════════
# TAB 3: ABOUT
# ════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("About This Project")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Dataset**
        - Heart Failure Clinical Records (Kaggle)
        - 299 patients from Faisalabad, Pakistan (2015)
        - 13 clinical features, binary outcome (DEATH_EVENT)
        - ~32% mortality rate in the dataset

        **Tech Stack**
        - Models: Scikit-learn, XGBoost
        - Explainability: SHAP (SHapley Additive exPlanations)
        - Frontend: Streamlit
        - Data: Pandas, NumPy, Seaborn
        """)

    with col2:
        st.markdown("""
        **Features in the Dataset**

        | Feature | Description |
        |---|---|
        | age | Patient age |
        | ejection_fraction | % blood pumped per beat |
        | serum_creatinine | Kidney function |
        | serum_sodium | Electrolyte balance |
        | platelets | Blood clotting cells |
        | CPK | Muscle damage enzyme |
        | anaemia | Low red blood cell count |
        | diabetes | Diabetes diagnosis |
        | high_blood_pressure | Hypertension |
        | smoking | Smoking status |
        | sex | Biological sex |
        | time | Follow-up period |
        """)

    st.subheader("What is SHAP?")
    st.markdown("""
    SHAP (SHapley Additive exPlanations) is a method from game theory that explains
    individual predictions. For each patient, SHAP calculates how much each feature
    pushed the prediction higher or lower compared to the average prediction.

    This is critical in medical ML — a model that just says "80% risk" isn't useful
    to a doctor. A model that says "80% risk because ejection fraction is critically
    low and serum creatinine is elevated" helps the doctor take specific action.
    """)

    st.subheader("SHAP Explainer Selection")
    st.markdown(f"""
    This app automatically selects the correct SHAP explainer based on the loaded model:

    | Model Type | SHAP Explainer Used |
    |---|---|
    | Random Forest, XGBoost | TreeExplainer (fastest, most accurate) |
    | Logistic Regression | LinearExplainer |
    | Any other model | KernelExplainer (universal fallback) |

    **Currently loaded:** `{model_name}`
    """)

    st.info(
        "This tool is for educational and research purposes only. "
        "It is not a substitute for clinical judgement or professional medical advice."
    )
