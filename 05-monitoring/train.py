"""Train Hospital Readmission Risk model and log to MLflow.

1. Load Diabetes 130-US Hospitals data
2. Patient-level split, feature engineering (per 02/03)
3. Fit Pipeline: DictVectorizer + XGBClassifier
4. Log to MLflow, write run_id.txt for app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "diabetic_data.csv"
EXPERIMENT_NAME = "hospital-readmission-monitoring"
SEED = 42


class DictVectorizerWrapper(BaseEstimator, TransformerMixin):
    """Sklearn-compatible wrapper around DictVectorizer."""

    def __init__(self):
        self.dv = DictVectorizer(sparse=True)

    def fit(self, X, y=None):
        self.dv.fit(X)
        return self

    def transform(self, X):
        return self.dv.transform(X)


FEATURE_COLS = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_emergency",
    "number_inpatient",
    "number_outpatient",
    "number_diagnoses",
    "care_intensity",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "age",
    "gender",
    "race",
    "change",
    "diabetesMed",
    "medication_changed",
    "A1Cresult",
    "max_glu_serum",
]


def read_data(path: Path, limit: int | None = 50_000) -> pd.DataFrame:
    """Load and preprocess Diabetes 130-US Hospitals data."""
    if not path.exists():
        raise FileNotFoundError(
            f"Data not found at {path}. "
            "Download from UCI (https://archive.ics.uci.edu/dataset/296) "
            "and place diabetic_data.csv in data/"
        )
    print("📥 Loading raw data ...")
    df = pd.read_csv(path)
    df["target"] = df["readmitted"].isin(["30", "<30"]).astype(int)
    if "weight" in df.columns:
        df = df.drop(columns=["weight"])
    for col in ["medical_specialty", "payer_code"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").replace("?", "Unknown")
    df["age"] = df["age"].fillna("[50-60)")
    df["gender"] = df["gender"].fillna("Unknown")
    df["change"] = df["change"].fillna("No")
    df["diabetesMed"] = df["diabetesMed"].fillna("No")
    for col in ["A1Cresult", "max_glu_serum"]:
        if col not in df.columns:
            df[col] = "not_tested"
        else:
            df[col] = df[col].fillna("None").replace("None", "not_tested").astype(str)
    if "race" in df.columns:
        df["race"] = df["race"].fillna("Unknown").replace("?", "Unknown").astype(str)
    for col in ["number_emergency", "number_inpatient", "number_outpatient"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0).astype(int)
    df["care_intensity"] = (
        df["number_emergency"] + df["number_inpatient"] + df["number_outpatient"]
    )
    df["medication_changed"] = (df["change"] == "Ch").astype(int)
    for col in ["num_lab_procedures", "num_procedures", "num_medications", "number_diagnoses"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0).astype(int)
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=SEED)
    print(f"✓ Loaded {len(df):,} rows, target positive rate: {df['target'].mean():.2%}")
    return df


def prepare_features(df: pd.DataFrame):
    """Create feature dicts and target."""
    cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[cols].copy()
    for col in ["age", "gender", "race", "change", "diabetesMed", "A1Cresult", "max_glu_serum"]:
        if col in X.columns:
            X[col] = X[col].astype(str)
    return X.to_dict(orient="records"), df["target"].values


def train_and_log(X_train, y_train, X_val, y_val):
    """Train model, log to MLflow, write run_id.txt."""
    print("🚀 Training model ...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    params = {
        "objective": "binary:logistic",
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "random_state": SEED,
        "eval_metric": "aucpr",
    }

    pipeline = Pipeline([
        ("vectorizer", DictVectorizerWrapper()),
        ("classifier", xgb.XGBClassifier(**params)),
    ])

    with mlflow.start_run() as run:
        pipeline.fit(X_train, y_train)
        y_pred_proba = pipeline.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, y_pred_proba)
        roc_auc = roc_auc_score(y_val, y_pred_proba)

        print(f"✓ PR-AUC: {pr_auc:.3f}  ROC-AUC: {roc_auc:.3f}")

        mlflow.log_params(params)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.sklearn.log_model(pipeline, "model")

        run_id = run.info.run_id
        with open("run_id.txt", "w") as f:
            f.write(run_id)
        print(f"💾 Saved run_id.txt (run: {run_id})")
        print(f"🖥  View MLflow UI: {MLFLOW_TRACKING_URI}")
        return run_id


def main():
    print("\n=== Hospital Readmission Risk Training ===\n")
    df = read_data(DATA_PATH)
    X, y = prepare_features(df)

    patient_target = df.groupby("patient_nbr")["target"].max()
    train_patients, val_patients = train_test_split(
        patient_target.index.tolist(),
        test_size=0.2,
        random_state=SEED,
        stratify=patient_target.values,
    )
    df_train = df[df["patient_nbr"].isin(train_patients)]
    df_val = df[df["patient_nbr"].isin(val_patients)]
    X_train, _ = prepare_features(df_train)
    X_val, y_val = prepare_features(df_val)
    y_train = df_train["target"].values

    train_and_log(X_train, y_train, X_val, y_val)
    print("\n✅ Training complete. Next: python app.py\n")


if __name__ == "__main__":
    main()
