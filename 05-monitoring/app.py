"""FastAPI service for Hospital Readmission Risk Prediction.

Loads Pipeline (DictVectorizer + XGBClassifier) from MLflow using run_id.txt.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
RUN_ID: Optional[str] = None
model = None


class PatientRequest(BaseModel):
    """Input payload for readmission risk prediction. Must match train.py FEATURE_COLS."""

    time_in_hospital: int = Field(..., ge=1, le=14, description="Days in hospital")
    num_lab_procedures: int = Field(..., ge=0, description="Number of lab procedures")
    num_procedures: int = Field(..., ge=0, description="Number of procedures")
    num_medications: int = Field(..., ge=0, description="Number of medications")
    number_emergency: int = Field(..., ge=0, description="Emergency visits (prior year)")
    number_inpatient: int = Field(..., ge=0, description="Inpatient visits (prior year)")
    number_outpatient: int = Field(..., ge=0, description="Outpatient visits (prior year)")
    number_diagnoses: int = Field(..., ge=0, description="Number of diagnoses")
    care_intensity: int = Field(..., ge=0, description="Sum of emergency+inpatient+outpatient")
    admission_type_id: int = Field(..., ge=1, description="Admission type ID")
    discharge_disposition_id: int = Field(..., ge=1, description="Discharge disposition ID")
    admission_source_id: int = Field(..., ge=1, description="Admission source ID")
    age: str = Field(..., description="Age group, e.g. [50-60)")
    gender: str = Field(..., description="Gender")
    race: str = Field(..., description="Race")
    change: str = Field(default="No", description="Medication change: No, Ch, etc.")
    diabetesMed: str = Field(default="No", description="Diabetes medication: Yes, No")
    medication_changed: int = Field(..., ge=0, le=1, description="1 if change==Ch else 0")
    A1Cresult: str = Field(default="not_tested", description="HbA1c result")
    max_glu_serum: str = Field(default="not_tested", description="Max glucose serum")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "time_in_hospital": 3,
                "num_lab_procedures": 41,
                "num_procedures": 0,
                "num_medications": 8,
                "number_emergency": 0,
                "number_inpatient": 0,
                "number_outpatient": 0,
                "number_diagnoses": 9,
                "care_intensity": 0,
                "admission_type_id": 1,
                "discharge_disposition_id": 1,
                "admission_source_id": 7,
                "age": "[50-60)",
                "gender": "Female",
                "race": "Caucasian",
                "change": "Ch",
                "diabetesMed": "Yes",
                "medication_changed": 1,
                "A1Cresult": "not_tested",
                "max_glu_serum": "not_tested",
            }
        }
    )


class PredictionResponse(BaseModel):
    risk_score: float
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global RUN_ID, model

    with open("run_id.txt", "r") as f:
        RUN_ID = f.read().strip()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(f"runs:/{RUN_ID}/model")
    print("[startup] Loaded Pipeline artifact 'model'.")
    yield


app = FastAPI(
    title="Hospital Readmission Risk Predictor",
    description="Predict 30-day readmission risk for diabetic inpatients.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"message": "Welcome to the Hospital Readmission Risk prediction API"}


@app.get("/health")
def health():
    return {"status": "ok", "run_id": RUN_ID}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    feature_dict = patient.model_dump()
    pred_proba = model.predict_proba([feature_dict])[0, 1]
    return PredictionResponse(
        risk_score=float(pred_proba),
        model_version=RUN_ID or "unknown",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=9696, reload=True)
