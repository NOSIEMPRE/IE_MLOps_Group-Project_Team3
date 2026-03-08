# Phase 1 — Data & EDA

**Milestones:** Data exploration, preprocessing, patient-level split, baseline (Logistic Regression)

## Contents

| Item | Description |
|------|-------------|
| `notebooks/EDA.ipynb` | Exploratory data analysis |
| `notebooks/EDA.md` | EDA summary and key findings |
| `notebooks/features.ipynb` | Preprocessing, feature engineering, patient-level split, saves `data/processed/` |

## Feature Engineering (`features.ipynb`)

1. **Preprocessing** — replace `?` with NaN, drop `weight`, filter deceased/hospice encounters, fill `medical_specialty`/`payer_code`/`race` missing as "Unknown"
2. **Drop near-zero-variance meds** — `examide`, `citoglipton`, `troglitazone`, `metformin-pioglitazone`, `glimepiride-pioglitazone`
3. **IDS code mapping** — `admission_type_id`, `discharge_disposition_id`, `admission_source_id` → interpretable categories
4. **ICD-9 → CCS categories** — 700+ codes per diagnosis slot collapsed to 14 clinical groups
5. **Charlson Comorbidity Index** — approximation from `diag_1/2/3`
6. **Medication features** — binary encoding + burden score + `change_flag` + `diabetes_med_flag`
7. **Demographic encoding** — `age_mid` (decade midpoint), `gender_male` (binary)
8. **Lab encoding** — `a1c_ordinal`, `glu_ordinal` (0–3), `a1c_tested`, `glu_tested` (binary)
9. **Aggregated features** — `total_prior_visits`, `log_prior_visits`, `has_prior_inpatient`, `care_intensity`
10. **Patient-level split** — 70/15/15 by `patient_nbr` (no patient overlap between splits)
11. **Save** → `data/processed/train.csv`, `val.csv`, `test.csv`, `features_full.csv`

## Next Steps

- [ ] Baseline — Logistic Regression on `data/processed/train.csv`, report PR-AUC, AUROC, Recall@K=20
