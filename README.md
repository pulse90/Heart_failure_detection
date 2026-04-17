# Heart Failure Detection System

A machine learning system that predicts risk of death from heart failure using
clinical patient data. Built with Scikit-learn, XGBoost, and SHAP explainability.

## Dataset
- **Source**: Heart Failure Clinical Records (Kaggle)
- **Link**: https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data
- **Size**: 299 patients, 13 features, binary target (DEATH_EVENT)

## Project Structure
```
heart-failure-detection/
├── data/
│   └── heart_failure_clinical_records.csv   ← download from Kaggle
├── notebooks/
│   └── eda.ipynb                            ← exploratory analysis
├── src/
│   ├── train.py                             ← trains & saves all models
│   └── predict.py                           ← loads model, makes prediction
├── app.py                                   ← Streamlit web app
├── model.pkl                                ← saved best model (auto-generated)
├── scaler.pkl                               ← saved scaler (auto-generated)
└── requirements.txt
```

## Setup & Run
```bash
pip install -r requirements.txt

# 1. Download dataset from Kaggle and place in data/ folder

# 2. Train models (saves best model as model.pkl)
python src/train.py

# 3. Launch the web app
streamlit run app.py
```

## Models Compared
| Model               | AUC-ROC | Recall | F1    |
|---------------------|---------|--------|-------|
| Logistic Regression | ~0.82   | ~0.72  | ~0.74 |
| Random Forest       | ~0.88   | ~0.76  | ~0.78 |
| XGBoost             | ~0.91   | ~0.80  | ~0.82 |

## Why Recall matters more than Accuracy
In medical diagnosis, a **false negative** (predicting a patient is safe when
they're actually at risk) is far more dangerous than a false positive.
We optimise for Recall to minimise missed high-risk patients.

## Key Features (by SHAP importance)
1. `serum_creatinine` — kidney function indicator
2. `ejection_fraction` — heart pumping efficiency
3. `time` — follow-up period length
4. `age` — patient age
5. `serum_sodium` — electrolyte balance

## Tech Stack
- **Models**: Scikit-learn, XGBoost
- **Explainability**: SHAP
- **Frontend**: Streamlit
- **Data**: Pandas, NumPy
- **Visualisation**: Matplotlib, Seaborn
