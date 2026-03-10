# MACHINE LEARNING OPERATIONS

## IE University — Group Project Checkpoint

# Hospital Readmission Risk Prediction: An End-to-End ML System for Proactive Patient Care

**Submission Date:** 8 March 2026  
**Dataset:** Diabetes 130-US Hospitals — UCI ML Repository, id 296 — Strack et al., 2014  
**Document:** Group Project Checkpoint Proposal — Revised  
**Team Members:** Marian, Marco, Yaxin, Lorenz, Jorge, and Omar

---

## Table of Contents

1. Executive Summary
2. Business Case & Objectives
3. Scope, Constraints & Risks
4. Data and Modeling Strategy
5. System-Level Considerations
6. References

---

## 1. Executive Summary

### 1.1 What problem are we solving?

We are building a machine learning system to predict 30-day readmission risk for diabetic inpatients. Under the US Hospital Readmissions Reduction Program (HRRP), hospitals face penalties of up to 3% of Medicare reimbursements when readmission rates exceed benchmarks. Diabetic patients have high readmission rates due to complex medication management, multiple comorbidities, and the need for careful post-discharge follow-up. This population is costly for hospitals but addressable through targeted intervention. Our system will provide a risk score at discharge, enabling care teams to identify which patients are most likely to return within 30 days and to allocate follow-up resources accordingly.

### 1.2 Why is it important?

Beyond financial penalties, readmissions signal care continuity failures that harm patients. When a patient returns to the hospital shortly after discharge, it often indicates that the transition from inpatient to outpatient care was inadequate. Identifying high-risk patients at discharge lets care teams intervene proactively before complications arise, for example through medication reconciliation, post-discharge phone calls, or early outpatient visits. This reduces avoidable readmissions and protects hospital revenue under value-based care. The HRRP has made readmission reduction a strategic priority for hospitals; our system supports that priority with data-driven risk stratification.

### 1.3 Who are the users or stakeholders?

Primary stakeholders include hospital finance teams, who are motivated to avoid HRRP penalties and to reduce the cost of avoidable readmissions; care coordinators, who would use risk scores daily to prioritize which patients receive follow-up; and attending physicians, who would rely on the system for discharge decision support. Secondary stakeholders are patients and their families, who benefit from smoother transitions of care and fewer disruptive readmissions. The system is designed to integrate into the discharge workflow so that risk scores and key clinical drivers are available when the care team makes decisions.

### 1.4 What outcome or impact do we expect?

We expect to reduce the 30-day readmission rate by at least 10% among flagged high-risk patients who receive care coordinator intervention. Intervention costs will remain below savings from avoided readmissions and HRRP penalties. Care teams will gain efficiency by prioritizing patients most likely to benefit from follow-up, rather than applying a one-size-fits-all approach. The system will be explainable: every risk score will be accompanied by a SHAP explanation so clinicians understand why a patient was flagged. We treat this project as a proof-of-concept on the Diabetes 130-US Hospitals dataset, with a path to deployment in a production EHR environment as a future step.

---

## 2. Business Case & Objectives

### 2.1 Business Objectives

We aim to create value by reducing avoidable readmissions, which improves patient outcomes and care continuity. Under the HRRP, hospitals face penalties of up to 3% of Medicare reimbursements when readmission rates exceed benchmarks (Centers for Medicare & Medicaid Services, 2023). For a hospital with substantial Medicare volume, this penalty can amount to millions of dollars annually. Our ROI target is to keep intervention costs below savings from avoided readmissions and these penalties. Cost reduction follows from fewer readmissions (each avoided readmission saves both direct care costs and opportunity costs) and penalty avoidance. For user impact, care coordinators will prioritize the highest-risk patients instead of spreading effort thinly across all discharges; physicians will receive discharge decision support that highlights which patients may need closer follow-up; and finance teams will protect revenue under value-based care by aligning clinical and financial incentives.

### 2.2 ML Objectives

Our ML objectives align with these business goals. For performance, we use AUROC for model selection across decision thresholds, Precision-Recall AUC for the minority class (readmitted patients), and Recall at K as the operational metric. Recall at K answers the question: when we flag the top K% of discharges as high-risk (e.g. K=20), how many of the actual readmitters do we catch? This metric reflects care coordinator capacity, since they can only intervene on a limited number of patients per day (Huyen, 2022). Latency is addressed through batch nightly pre-computation: scores are computed in advance so they are ready when the discharge decision is made, without requiring real-time inference. Every risk score includes a SHAP explanation for interpretability, so clinicians understand which factors drove the prediction (Lundberg & Lee, 2017). Fairness is enforced by checking performance across demographics; a 5% gap in recall or false positive rate between subgroups blocks deployment (Huyen, 2022; IE University, 2026).

### 2.3 How ML Objectives Support the Business Case

These ML objectives directly support the business case. PR-AUC and Recall at K ensure we identify the patients most likely to benefit from intervention; a model that optimizes for these metrics will surface the right patients for care coordinators to contact. When care coordinators act on these high-risk flags, readmissions drop and ROI and cost reduction follow. Nightly batch scoring keeps latency acceptable without real-time infrastructure, reducing operational complexity and cost. SHAP explanations build clinician trust and support discharge decisions: a physician can see, for example, that a patient was flagged due to high prior emergency visits and medication changes, and can tailor the discharge plan accordingly. Fairness checks protect vulnerable subgroups from biased risk scores, which is both ethically necessary and operationally important, since biased models could lead to inequitable allocation of follow-up resources.

### 2.4 Success Criteria

Our success criteria are threefold. The target metric is at least 10% reduction in 30-day readmission rate among flagged high-risk patients who receive care coordinator intervention. This is a conservative but achievable goal given the literature on post-discharge intervention effectiveness. The business KPI is that intervention costs remain below savings from avoided readmissions and HRRP penalties; we will not deploy a system that increases net cost. The ML gate is that a model must exceed the Logistic Regression baseline on validation PR-AUC to be promoted; we do not deploy a more complex model unless it demonstrably outperforms the simpler baseline.

---

## 3. Scope, Constraints & Risks

### 3.1 Scope

The project focuses on 30-day readmission prediction for diabetic inpatients using the Diabetes 130-US Hospitals dataset. We build a batch scoring system with nightly pre-computation and SHAP explanations. The system produces risk scores and key clinical drivers for every diabetic discharge, consumed at discharge workflow lookup and the care coordinator dashboard. Fairness is monitored across race and payer type, and experiments are tracked in MLflow with CI/CD on GitHub. We target MLOps Level 1 maturity, with Level 2 as a stretch goal. Real-time API, federated learning, and production EHR deployment are out of scope for this checkpoint; we design the pipeline so that these could be added in a future phase.

### 3.2 Constraints

We work with the Diabetes 130-US Hospitals dataset, which covers 1999–2008 only; there is no live EHR or external validation cohort available for this project. The project timeline is approximately six weeks, which limits the depth of experimentation and the number of model variants we can test. Compute can be local or cloud; we do not assume access to GPU or large-scale distributed training. Labels are pre-defined in the dataset, so no manual labeling is required. The dataset is public and de-identified, which simplifies data governance but means we cannot validate predictions against real-world outcomes in a production setting during this phase.

### 3.3 Risks

We address data imbalance (11.2% positive class) through stratified splits, SMOTE on the training set only, and PR-AUC as the primary metric. Small subgroups (e.g. Asian, Hispanic) may yield noisy fairness metrics due to limited sample sizes, so we report confidence intervals and do not over-interpret subgroup differences. Data staleness is a concern given the 1999–2008 range; clinical practice and coding have evolved since then, so we treat this project as a proof-of-concept and do not claim generalizability to current hospital populations without further validation. The 10% readmission reduction target applies to *intervened* patients only, not the full cohort; we measure impact among those who receive care coordinator follow-up, since the model's value is realized through intervention.

---

## 4. Data and Modeling Strategy

### 4.1 Data Plan

The Diabetes 130-US Hospitals dataset (UCI ML Repository, id 296; Strack et al., 2014) is a public dataset extracted from Cerner's Health Facts database. It contains 101,766 inpatient diabetic encounters from 130 US hospitals (1999–2008), with 50 features per encounter.

EDA reveals an 11.2% positive class and 71,518 unique patients, of whom 16,773 have multiple encounters. Demographics skew older (64.7% aged 50–80) and predominantly Caucasian (74.8%); readmission rate is roughly 10–11% across race and gender. Asian and Hispanic subgroups may yield noisy fairness metrics due to small sample sizes. For completeness, we drop *weight* (96.9% missing), treat *medical_specialty* (49.1% missing) and *payer_code* (39.6% missing) as "Unknown," and treat HbA1c "None" (83.3%) as informative "not tested."

Labels are pre-defined in the dataset; no manual labeling is required. The *readmitted* column has three classes; we collapse to binary (readmitted within 30 days or not) because 30 days is the HRRP window. Given the 11.2% positive class, we use stratified splits by *patient_nbr* (train 70%, validation 15%, test 15%) to preserve balance and avoid leakage. SMOTE is applied on training only, and the decision threshold is tuned to care coordinator capacity.

We avoid several common pitfalls. The patient-level split prevents leakage across the 16,773 patients with multiple encounters. PR-AUC as the primary metric avoids the accuracy trap under class imbalance. There is no label multiplicity; we map ICD-9 codes to CCS groupings to reduce cardinality (Strack et al., 2014).

### 4.2 Baselines

Logistic Regression serves as our baseline: calibrated and interpretable, it sets the minimum performance bar. Our usefulness threshold is that a model must exceed the baseline on validation PR-AUC to be considered for promotion. We also establish a "predict no readmission" baseline (89% accuracy) to illustrate why accuracy alone is misleading under class imbalance.

### 4.3 Candidate Models

We test XGBoost and LightGBM first because they are standard for structured tabular clinical data and handle class imbalance natively (Huyen, 2022). We use Optuna for Bayesian hyperparameter search over learning rate, max depth, number of estimators, sub-sample ratio, and class weight. A model is promoted only if it exceeds the current champion on held-out validation PR-AUC. MLflow tracks all experiments.

### 4.4 Feature Engineering Ideas

| Feature category        | Description / details                                                                 |
| ----------------------- | ------------------------------------------------------------------------------------- |
| Demographics            | Age (10-year bins), gender, race                                                      |
| Admission context       | Admission type, discharge disposition, admission source; mapped via IDS codes         |
| Clinical utilization    | Counts of lab procedures, medications, procedures, diagnoses; prior inpatient, outpatient, emergency visits in preceding year (heavily right-skewed) |
| Diagnoses               | Primary, secondary, tertiary ICD-9 codes; 700+ unique values per slot; mapped to CCS groupings (~9 categories); target-encode top predictive groups |
| Lab results             | HbA1c and max glucose serum; each has four possible values                           |
| Medications             | 23 binary or categorical change flags; insulin 53.4%, metformin, glipizide, glyburide; five near-zero-variance columns dropped |
| Medication change       | Whether any medication was changed; whether any diabetes medication was prescribed    |
| Engineered (planned)    | Charlson comorbidity approximation, medication burden score, care intensity composite |
| Dropped                 | *weight* (96.9% missing); five near-zero-variance medication columns                  |

### 4.5 Evaluation Metrics

We use AUROC for model selection across thresholds. Precision-Recall AUC is more honest for the minority class under imbalance. Recall at K serves as the operational metric: it measures how many high-risk patients we catch when flagging the top K% of discharges (e.g. K=20), reflecting care coordinator capacity (Huyen, 2022). PR-AUC is the primary objective for hyperparameter search and model promotion.

### 4.6 Interpretability

The system must be explainable from the start. Every risk score is accompanied by a SHAP explanation so clinicians know why a patient was flagged (Lundberg & Lee, 2017). We use tree-based models (XGBoost/LightGBM) with SHAP rather than black-box ensembles to maintain interpretability while achieving strong performance.

---

## 5. System-Level Considerations

### 5.1 Serving Architecture

We use batch prediction: nightly scoring of active diabetic inpatients, with scores stored for lookup at discharge. Care coordinators receive a prioritized high-risk list. This approach is simpler and more reliable than a real-time API (Huyen, 2022). Outputs are consumed at discharge workflow lookup and the care coordinator dashboard.

### 5.2 Pipeline and Versioning

The pipeline comprises ingestion, validation, feature engineering, training, evaluation, and deployment. Code is version-controlled in GitHub, and CI/CD (GitHub Actions) runs tests on pull requests. A lightweight scheduler (e.g. Prefect) orchestrates nightly runs. Shared feature definitions prevent training-serving skew (Huyen, 2022).

### 5.3 Monitoring and Drift Detection

We monitor three layers of drift: data drift (PSI on inputs; GeeksForGeeks, 2025), prediction drift (output score distribution), and outcome drift (30-day readmission rate with lag; Huyen, 2022). When drift is detected, we trigger retraining on a rolling 12-month window with a validation gate. Our target is MLOps Level 1, with Level 2 as a stretch goal.

### 5.4 Fairness Monitoring

Recall and false positive rate are monitored by race and payer. A 5% demographic gap blocks deployment. A fairness review is required before production. Note that Asian and Hispanic subgroups may yield noisy metrics due to small sample sizes.

### 5.5 System Preview & Project Plan

**Data flow**

```
TRAINING PIPELINE
─────────────────────────────────────────────────────────────────────────
data/diabetic_data.csv
        │
        ▼
   Ingestion & Validation (patient-level split 70/15/15, missing handling, IDS mapping)
        │
        ▼
   Feature Engineering (CCS groupings, Charlson, medication burden, drop weight/low-variance meds)
        │
        ▼
   Model Training (XGBoost / LightGBM) ──► Optuna ──► MLflow
        │
        ▼
   Evaluation (PR-AUC, fairness) ──► Validation Gate ──► Model Registry
        │
        │
INFERENCE (nightly, Prefect)
─────────────────────────────────────────────────────────────────────────
        │
        ▼
   Active inpatients ──► Feature Engineering ──► Model ──► Risk Score + SHAP
        │
        ├──────────────────► Discharge Workflow Lookup
        │
        └──────────────────► Care Coordinator Dashboard
                │
                └──► Drift Detection ──► Retrain if triggered
```

Raw data is ingested and validated (Section 4.1), then passes to feature engineering (Section 4.4). The training pipeline uses Optuna for hyperparameter search and MLflow for tracking; a validation gate ensures the model exceeds the baseline on PR-AUC and passes fairness checks before deployment. Nightly batch inference (orchestrated by Prefect) produces risk scores with SHAP explanations for discharge lookup and the care coordinator dashboard. Drift detection triggers retraining when needed.

**Timeline & milestones**


| Milestone       | Timeline | Description                                                                 |
| --------------- | -------- | --------------------------------------------------------------------------- |
| Data collection | 1–2      | Load and validate diabetic dataset; EDA; patient-level split; preprocessing |
| Baseline        | 3–4      | Logistic Regression; establish minimum performance bar                      |
| First model     | 3–4      | XGBoost/LightGBM training; Optuna tuning; model selection                   |
| Evaluation      | 5–6      | Fairness evaluation; SHAP explanations; validation gate                     |
| Final demo      | 5–6      | Pipeline deployment; monitoring setup; documentation; demo presentation     |


**Team roles**


| Role             | Responsibility                                        | Team Member(s) |
| ---------------- | ----------------------------------------------------- | -------------- |
| Data & EDA       | Data exploration, preprocessing, split strategy       | TBD            |
| Features         | Feature engineering, CCS mapping, engineered features | TBD            |
| Modeling         | Model training, hyperparameter tuning, evaluation     | TBD            |
| Pipeline & MLOps | CI/CD, MLflow, deployment, monitoring                 | TBD            |
| Documentation    | Proposal, README, final report                        | TBD            |


*Note: Roles can overlap; all areas covered by Marian, Marco, Yaxin, Lorenz, Jorge, Omar.*

---

## 6. References

Centers for Medicare & Medicaid Services. (2023). *Hospital Readmissions Reduction Program (HRRP)*. [https://www.cms.gov/medicare/quality-initiatives-patient-assessment-instruments/value-based-programs/hrrp/hospital-readmission-reduction-program](https://www.cms.gov/medicare/quality-initiatives-patient-assessment-instruments/value-based-programs/hrrp/hospital-readmission-reduction-program)

GeeksForGeeks. (2025, August 13). *Population Stability Index*. [https://www.geeksforgeeks.org/data-science/population-stability-index-psi/](https://www.geeksforgeeks.org/data-science/population-stability-index-psi/)

Huyen, C. (2022). *Designing Machine Learning Systems*. O'Reilly Media.

IE University. (2026). *Machine Learning Operations* (Course Sessions 1–5).

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, *30*, 4765–4774.

Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). Impact of HbA1c measurement on hospital readmission rates: Analysis of 70,000 clinical database patient records. *BioMed Research International*, *2014*, Article 781670. [https://doi.org/10.1155/2014/781670](https://doi.org/10.1155/2014/781670)

UCI Machine Learning Repository. (2014). *Diabetes 130-US hospitals dataset*. [https://doi.org/10.24432/C5230J](https://doi.org/10.24432/C5230J)