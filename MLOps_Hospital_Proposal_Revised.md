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
2. Business Problem  
3. Clear Objectives  
4. Scope, Constraints & Risks  
5. Data and Modeling Strategy  
6. System-Level Considerations  
7. References  

---

## 1. Executive Summary

**What problem are we solving?**  
We are building a machine learning system to predict 30-day readmission risk for diabetic inpatients. Under the US Hospital Readmissions Reduction Program—HRRP—, hospitals face financial penalties of up to 3% of Medicare reimbursements when readmission rates exceed benchmarks. Diabetic patients have particularly high readmission rates due to complex medication management and comorbidities, making this population both costly and addressable through targeted intervention.

**Why is it important?**  
Beyond financial penalties, readmissions signal failures in care continuity that directly harm patients. By identifying high-risk patients at discharge, care teams can intervene before complications arise—reducing avoidable readmissions, improving patient outcomes, and protecting hospital revenue under value-based care models.

**Who are the users or stakeholders?**  
Primary stakeholders include hospital finance teams, who seek to avoid HRRP penalties; care coordinators, who use risk scores daily to prioritize follow-up; and attending physicians, who rely on discharge decision support. Secondary stakeholders include patients and their families, who benefit from smoother transitions of care.

**What outcome or impact do we expect?**  
We expect to reduce the 30-day readmission rate by at least 10% among flagged high-risk patients who receive care coordinator intervention. The cost of targeted interventions—such as follow-up calls and care-coordinator time—will remain below the financial savings from avoided readmissions and HRRP penalties. Care teams will gain efficiency by prioritizing the patients most likely to benefit from follow-up, given constrained clinical capacity.

---

## 2. Business Problem

### 2.1 Context

Hospitals operating under value-based care models face financial penalties if too many of their patients get readmitted within 30 days of being discharged. In the US, the Hospital Readmissions Reduction Program—HRRP—penalizes hospitals up to 3% of all Medicare reimbursements (Centers for Medicare & Medicaid Services, 2023). Diabetic patients have high readmission rates because they have complicated health situations and tricky medication management, which makes this cost a lot of money to hospitals, but it is something addressable. Beyond finances, readmissions are a sign of failures in care continuity that directly harm patients.

### 2.2 Proposed Solution

We want to build a Machine Learning system that gives the care team a risk score at the point of discharge, indicating whether a patient is likely to be readmitted within 30 days. Scores are pre-computed nightly on all active diabetic inpatients, ensuring they are ready and available when the discharge decision is made, enabling doctors to intervene with targeted follow-up before the patient leaves the hospital. The system will integrate into the discharge workflow, providing a risk score and its key clinical drivers for every diabetic discharge event.

### 2.3 Business Objectives

Our objectives are threefold. First, we aim to reduce the 30-day readmission rate by at least 10% among flagged high-risk patients who receive care coordinator intervention. Second, we will ensure that the cost of targeted interventions—such as follow-up calls and care-coordinator time—remains below the financial savings from avoided readmissions and HRRP penalties. Third, we seek to improve care teams' efficiency by prioritizing the patients most likely to benefit from follow-up, given constrained clinical capacity. Primary stakeholders include hospital finance teams focused on penalty avoidance, care coordinators as daily users of the risk score, and attending physicians for discharge decision support.

---

## 3. Clear Objectives

### 3.1 ML Task

Binary classification: given a patient's clinical record at discharge, predict 30-day readmission—yes or no. The original dataset has three possible outcomes: <30 days, >30 days, and no readmission. We will collapse that into just two: readmitted within 30 days or not readmitted. We are doing this because 30 days is the window that matters for the HRRP penalty, and it is what the care coordinators and teams can make decisions with.

### 3.2 Evaluation Metrics

Given approximately 11% class imbalance, if the model simply predicted "nobody gets readmitted", it would be 89% accurate—which is useless. We will use three metrics. AUROC serves as our model selection metric, measuring how well the model ranks high-risk patients above low-risk ones across all possible thresholds when comparing models against each other. Precision-Recall AUC is similar to AUROC but focuses specifically on the minority class—our target variable—making it more honest for imbalanced data. Recall at K is our operational metric: of all patients who are truly high-risk, how many does the model catch when we can only flag the top K% of discharges per day—for example, K = 20—reflecting realistic care-coordinator capacity (Huyen, 2022).

### 3.3 System Requirements

The system must meet five requirements. Reliability means the system should continue to perform the correct function at the desired level of performance regardless of adversity. Maintainability is achieved through versioned models and pipeline code, with reproducible runs via MLflow experiment tracking. Scalability requires that the batch pipeline handle growing patient volumes—such as onboarding additional hospitals—without manual re-engineering. Interpretability is ensured by providing every risk score with an explanation using SHAP values, so doctors know why a patient was flagged (Lundberg & Lee, 2017). Fairness is enforced by checking the model's performance across demographics and socioeconomic groups; a gap exceeding 5% will block deployment (Huyen, 2022; IE University, 2026).

---

## 4. Scope, Constraints & Risks

### 4.1 In-Scope

The project scope includes 30-day readmission prediction for diabetic inpatients using the Diabetes 130-US Hospitals dataset, batch scoring with nightly pre-computation of risk scores, SHAP-based explanations for every risk score, fairness monitoring across race and payer type, MLflow experiment tracking and model versioning, and a CI/CD pipeline with GitHub Actions.

### 4.2 Out-of-Scope

We explicitly exclude a real-time inference API, as batch scoring is sufficient for the discharge workflow. Multi-hospital federated learning, clinical decision support system integration—we provide risk scores while clinical decisions remain with physicians— and deployment to production EHR systems are also out of scope for this proof-of-concept.

### 4.3 Constraints

We are constrained by data availability—1999–2008 data only, with no access to live EHR or external validation datasets—by time—the course timeline of approximately 6 weeks—by compute—local or cloud resources available to the team—and by labeling—labels are pre-defined in the dataset with no manual labeling required.

### 4.4 Risks

Several risks warrant attention. Data imbalance—11.2% positive—is mitigated through stratified splits, SMOTE on training only, and PR-AUC as the primary metric. Small subgroups—Asian: 641, Hispanic: 2,037—may produce noisy fairness metrics; we will report confidence intervals and avoid over-interpretation. Data staleness is a concern, as 1999–2008 data may not reflect current clinical practice; we treat this as a proof-of-concept and discuss generalizability in monitoring. Regarding unrealistic expectations, we set a 10% reduction target among *intervened* patients, not overall; success depends on care coordinator follow-through.

### 4.5 Trade-offs

We face two main trade-offs. First, recall vs. precision: given limited care coordinator capacity, we prioritize recall—catching high-risk patients—over precision and tune the threshold to flag the top K% of discharges. Second, interpretability vs. model complexity: we use SHAP with tree-based models—XGBoost or LightGBM—to maintain interpretability while achieving strong performance, and we avoid black-box ensembles that sacrifice explainability.

---

## 5. Data and Modeling Strategy

### 5.1 Dataset

The dataset used in this project is the Diabetes 130-US Hospitals dataset from UCI ML Repository, id 296 (UCI Machine Learning Repository, 2014), introduced by Strack et al. (2014). It was extracted from the Health Facts database—Cerner Corporation—a national clinical data warehouse that covers 74 million unique encounters across 17 million unique patients. The final dataset has 101,766 inpatient diabetic encounters from 130 US hospitals and integrated delivery networks from 1999 to 2008, with 50 features per encounter. Only inpatients with diabetic admissions of 1–14 days duration with laboratory tests performed and medications administered are included. Encounters resulting in patient death or transfer were excluded to ensure readmission was a possible outcome. A known limitation is that the data spans 1999–2008; clinical protocols and coding practices have since evolved. We treat this dataset as a proof-of-concept and discuss generalizability in Section 6.3.

An initial exploratory data analysis reveals several properties that directly shape our preprocessing, splitting, and modeling decisions. The original *readmitted* column has three classes: NO—54,864, 53.9%—, >30 days—35,545, 34.9%—, and <30 days—11,357, 11.2%. For our binary task, the positive class—<30 days—represents 11.2% of encounters, confirming substantial class imbalance. The 101,766 encounters correspond to only 71,518 unique patients, with 16,773 patients appearing in multiple encounters—up to 40 per patient. A naive random split would leak patient-level information between training and test sets, so we adopt a patient-level split strategy in Section 5.4. Demographically, the cohort skews older—64.7% of encounters are aged 50–80—and is predominantly Caucasian at 74.8%, with African American patients comprising 18.9%. Gender is roughly balanced—53.8% female, 46.2% male. The 30-day readmission rate is fairly uniform across race—10–11%—and gender—approximately 11%— which provides a reasonable baseline for fairness evaluation, though small sample sizes for Asian—641—and Hispanic—2,037—subgroups may produce noisy fairness estimates.

### 5.2 Key Features

Our feature set includes demographics—age in 10-year bins, gender, race—admission context—admission type, discharge disposition, admission source mapped via IDS codes—and clinical utilization—number of lab procedures with mean 43.1, number of medications with mean 16.0, number of procedures with mean 1.3, number of diagnoses with mean 7.4, and prior inpatient/outpatient/emergency visits in the preceding year, heavily right-skewed with most patients having 0. Diagnoses are represented by primary, secondary, and tertiary ICD-9 codes—717–790 unique codes each—mapped to CCS groupings of approximately nine clinical categories. Lab results include HbA1c with four values—None, Norm, >7, >8—and max glucose serum with four values—None, Norm, >200, >300. Medications are captured by 23 binary/categorical change flags; insulin dominates at 53.4% of encounters, followed by metformin at 19.6%, glipizide at 12.5%, and glyburide at 10.5%. Five medications have near-zero variance—examide, citoglipton, troglitazone, metformin-pioglitazone, glimepiride-pioglitazone—and will be dropped. Medication change indicators include *change*—whether any medication was changed, at 46.2% yes—and *diabetesMed*—whether any diabetes medication was prescribed, at 77.0% yes. We also plan to engineer a comorbidity index—Charlson approximation—medication burden score, and care intensity composite. The *weight* column has 96.9% missing values—98,569 of 101,766 rows—and is essentially unusable; we will drop it from the feature set.

### 5.3 Data Challenges and Mitigations

We address several data challenges as follows. For class imbalance—11.2% positive—we use stratified splits, SMOTE on training only, PR-AUC as the primary metric, and a threshold tuned to clinical capacity. To avoid patient-level data leakage—16,773 patients with multiple encounters—we split by *patient_nbr* rather than by encounter, ensuring no patient appears in both train and test sets. For HbA1c missingness—83.3%—we treat "None" as an informative "not tested" category rather than imputing a clinical value. The *weight* column is 96.9% missing and is dropped entirely. For *medical_specialty*—49.1% missing—and *payer_code*—39.6% missing—we treat missing values as an "Unknown" category to avoid imputation that could introduce bias; when *payer_code* is used for fairness monitoring, we note that payer subgroups may have small samples. For ICD-9 high cardinality—700+ unique codes per diagnosis slot (Strack et al., 2014)—we map to CCS groupings of approximately nine categories instead of one-hot-encoding and target-encode top predictive groups. Because actual outcomes arrive with a 30-day delay, we monitor data and prediction drift in the meantime.

### 5.4 Modeling Approach

We use Logistic Regression as our baseline—calibrated and interpretable, it sets the minimum performance bar. Our primary models are XGBoost and LightGBM, which are standard for structured tabular clinical data and handle class imbalance natively (Huyen, 2022). We will split by *patient_nbr* into train—70%—, validation—15%—, and test—15%—sets, ensuring no patient overlap across splits. Stratification on the binary target preserves the 11.2% positive rate in each subset.

We plan to use Optuna for Bayesian hyperparameter search over key parameters—learning rate, max depth, number of estimators, sub-sample ratio, class weight or scale_pos_weight. Each trial will be evaluated on the validation set using PR-AUC as the primary objective.

All experiments will be tracked in MLflow—hyperparameters, AUROC, PR-AUC, F1 at multiple thresholds. A model is promoted to the registry only if it exceeds the current champion on the held-out validation set.

---

## 6. System-Level Considerations

### 6.1 Serving Architecture

Batch prediction is the appropriate serving pattern. Every night, the system scores all active diabetic inpatients and those flagged as likely to be discharged soon. Risk scores are stored and made available for lookup, so they are ready when the discharge decision is made the following day. Care coordinators also receive a prioritized list of high-risk patients for post-discharge follow-up planning. This approach is simpler, cheaper, and more reliable than building a real-time inference API (Huyen, 2022).

Predictions are consumed in two places: the discharge workflow lookup, where risk score and SHAP explanation are available at the point of discharge, and the care coordinator dashboard, which displays a prioritized list of high-risk patients for follow-up planning.

### 6.2 Pipeline and Versioning

The full pipeline includes ingestion, validation, feature engineering, training, evaluation, and deployment. Pipeline code and configuration are version-controlled in GitHub. A CI/CD workflow—GitHub Actions—runs unit and integration tests on every pull request. For pipeline orchestration, we plan to use a lightweight scheduler—such as GitHub Actions scheduled workflows or Prefect—to coordinate nightly batch runs. Feature definitions are shared between training and inference pipelines to prevent training-serving skew, a common production failure mode (Huyen, 2022).

### 6.3 Monitoring and Drift Detection

Patient populations are not stationary: COVID-19, seasonal illness cycles, and clinical protocol changes can all shift the feature distribution and the label relationship. We implement three monitoring layers. Data drift is tracked via PSI—Population Stability Index—on key input features to detect whether input distributions are changing (GeeksForGeeks, 2025). Prediction drift monitors the distribution of output risk scores for unexpected shifts. Outcome drift tracks the actual 30-day readmission rate with a 30-day lag as actual outcomes feedback, to detect whether the real readmission rate is changing (Huyen, 2022). Drift alerts trigger automated retraining on a rolling 12-month window, with a validation gate before promotion. Our target is MLOps Maturity Level 1—automated CT and CD with passive monitoring—as baseline, with Level 2—active auto-trigger—as a stretch goal.

### 6.4 Fairness Monitoring

To avoid potential bias, recall and False Positive Rate are continuously monitored, separated by race and payer type. A demographic gap exceeding 5 percentage points in recall or False Positive Rate triggers an automatic deployment block. The model must pass a fairness review before promotion to production. Note: Asian and Hispanic subgroups have small sample sizes—641 and 2,037 encounters respectively; fairness metrics for these groups may be noisy and should be interpreted with caution.

### 6.5 System Preview & Project Plan

**Block diagram — data flow:**

```
diabetic_data.csv
        │
        ▼
   Preprocessing — validation, missing handling, IDS mapping
        │
        ▼
   Feature Engineering — CCS groupings, engineered features, drop weight/low-variance meds
        │
        ▼
   Model — XGBoost / LightGBM
        │
        ▼
   Risk Score + SHAP Explanations
        │
        ├──────────────────► Discharge Workflow Lookup
        │
        └──────────────────► Care Coordinator Dashboard
```

**Timeline & milestones:**

| Phase | Weeks | Milestones |
|-------|-------|------------|
| Data & EDA | 1–2 | Data exploration, baseline—Logistic Regression—, and patient-level split validation |
| Modeling | 3–4 | XGBoost/LightGBM training, Optuna tuning, model selection, fairness evaluation |
| Deployment & Docs | 5–6 | Pipeline deployment, monitoring setup, documentation, final demo |

**Team roles:**

| Role | Responsibility | Team Member(s) |
|------|----------------|---------------|
| Data & EDA | Data exploration, preprocessing, split strategy | TBD |
| Features | Feature engineering, CCS mapping, engineered features | TBD |
| Modeling | Model training, hyperparameter tuning, evaluation | TBD |
| Pipeline & MLOps | CI/CD, MLflow, deployment, monitoring | TBD |
| Documentation | Proposal, README, final report | TBD |

*Note: Roles can overlap; all areas should be covered by the team—Marian, Marco, Yaxin, Lorenz, Jorge, Omar.*

---

## 7. References

Centers for Medicare & Medicaid Services. (2023). *Hospital Readmissions Reduction Program (HRRP)*. https://www.cms.gov/medicare/quality-initiatives-patient-assessment-instruments/value-based-programs/hrrp/hospital-readmission-reduction-program

GeeksForGeeks. (2025, August 13). *Population Stability Index*. https://www.geeksforgeeks.org/data-science/population-stability-index-psi/

Huyen, C. (2022). *Designing Machine Learning Systems*. O'Reilly Media.

IE University. (2026). *Machine Learning Operations* (Course Sessions 1–5).

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, *30*, 4765–4774.

Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). Impact of HbA1c measurement on hospital readmission rates: Analysis of 70,000 clinical database patient records. *BioMed Research International*, *2014*, Article 781670. https://doi.org/10.1155/2014/781670

UCI Machine Learning Repository. (2014). *Diabetes 130-US hospitals dataset*. https://doi.org/10.24432/C5230J
