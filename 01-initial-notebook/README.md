# 01 – Initial Notebook

**Hospital Readmission Risk Prediction** — Exploratory Data Analysis

## Overview

This module contains the initial exploratory data analysis (EDA) for the Diabetes 130-US Hospitals dataset. The notebook examines data structure, missing values, target distribution, and key clinical features.

## Contents

| File | Description |
| :--- | :--- |
| `hospital_readmission_prediction.ipynb` | **Initial notebook**: Load → Prepare features → Train (Logistic Regression) → Evaluate |
| `eda_diabetes_readmission.ipynb` | EDA notebook: data loading, summary stats, visualizations |

## Data Path

- `../data/diabetic_data.csv` — Diabetes 130-US Hospitals dataset (UCI ML Repository, id 296)
- `../data/IDS_mapping.csv` — Admission/discharge ID mapping

## Key Findings

- **101,766 encounters**, 71,518 unique patients — patient-level split required to avoid leakage
- **11.2%** positive class (30-day readmission) — class imbalance
- **weight** 96.9% missing — to be dropped
- **HbA1c** 83.3% "not tested" — treat as informative category
- **medical_specialty** 49.1%, **payer_code** 39.6% missing — treat as "Unknown" category
