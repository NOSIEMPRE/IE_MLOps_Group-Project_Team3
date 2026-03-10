# 03 – Experiment Tracking

**Hospital Readmission Risk Prediction** — MLflow experiment tracking

## Overview

This module uses MLflow for experiment tracking, model versioning, and metrics logging. Per proposal 4.2–4.5: Logistic Regression is the baseline; XGBoost/LightGBM are tuned with Optuna. A model is promoted only if validation PR-AUC exceeds the baseline.

## Contents

| File | Description |
| :--- | :--- |
| `scenario-1.ipynb` | Baseline (Logistic Regression) — local SQLite (`mlflow.db`) |
| `scenario-2.ipynb` | XGBoost/LightGBM + Optuna — connects to scenario-1 UI (port 5001) |

## How to Run

### Scenario 1 (Baseline)
1. Ensure `../data/diabetic_data.csv` exists.
2. Run all cells in `scenario-1.ipynb`.
3. Baseline metrics (example): PR-AUC ≈ 0.189, ROC-AUC ≈ 0.644, Recall@K20 ≈ 0.35.
4. **Launch MLflow UI** (last cell) — http://localhost:5001. Keep it running for scenario-2.

### Scenario 2 (Optuna + Champion)
1. **Run scenario-1 first** and launch the MLflow UI (port 5001) — scenario-2 connects to it.
2. Run all cells in `scenario-2.ipynb`. `BASELINE_PR_AUC` is set to 0.189 (from scenario-1).
3. Champion is registered to Model Registry only if PR-AUC > baseline.

## Metrics

- **PR-AUC** — Primary metric for model selection (class imbalance)
- **Recall at K** — Operational metric (care coordinator capacity; K=20%)
- **ROC-AUC** — Model selection across thresholds
