"""
generate_data.py
Generates a synthetic heart failure dataset that mirrors the real Kaggle dataset.
Use this for development if you haven't downloaded the Kaggle dataset yet.

Real dataset: https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data
Run: python src/generate_data.py
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 299

age              = np.random.normal(60, 12, n).clip(40, 95).astype(int)
ejection_frac    = np.random.normal(38, 12, n).clip(14, 80).astype(int)
serum_creatinine = np.round(np.random.exponential(1.2, n).clip(0.5, 9.4), 1)
serum_sodium     = np.random.normal(136, 4, n).clip(113, 148).astype(int)
platelets        = np.round(np.random.normal(263000, 97000, n).clip(25000, 850000), 0)
cpk              = np.random.exponential(600, n).clip(23, 7861).astype(int)
time             = np.random.randint(4, 285, n)

anaemia          = np.random.binomial(1, 0.43, n)
diabetes         = np.random.binomial(1, 0.42, n)
high_bp          = np.random.binomial(1, 0.35, n)
sex              = np.random.binomial(1, 0.65, n)
smoking          = np.random.binomial(1, 0.32, n)

# Target: higher risk with low ejection fraction, high creatinine, older age
risk_score = (
    - 0.03 * ejection_frac
    + 0.5  * serum_creatinine
    + 0.02 * age
    - 0.02 * time
    + 0.3  * anaemia
    + np.random.normal(0, 0.5, n)
)
prob_death   = 1 / (1 + np.exp(-risk_score + 1.5))
death_event  = (prob_death > 0.5).astype(int)

df = pd.DataFrame({
    "age":                        age,
    "anaemia":                    anaemia,
    "creatinine_phosphokinase":   cpk,
    "diabetes":                   diabetes,
    "ejection_fraction":          ejection_frac,
    "high_blood_pressure":        high_bp,
    "platelets":                  platelets,
    "serum_creatinine":           serum_creatinine,
    "serum_sodium":               serum_sodium,
    "sex":                        sex,
    "smoking":                    smoking,
    "time":                       time,
    "DEATH_EVENT":                death_event,
})

os.makedirs("data", exist_ok=True)
df.to_csv("data/heart_failure_clinical_records.csv", index=False)
print(f"Dataset saved: data/heart_failure_clinical_records.csv")
print(f"Shape: {df.shape}")
print(f"Death rate: {df['DEATH_EVENT'].mean():.1%}")
