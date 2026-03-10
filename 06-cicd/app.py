"""FastAPI service for Hospital Readmission Risk Prediction.

Model baked into Docker image: models/model/, run_id.txt.
Aligns with 04/05: full FEATURE_COLS schema.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


RUN_ID: Optional[str] = None
model: Optional[Any] = None


class PatientRequest(BaseModel):
    """Request body for /predict. Must match train.py FEATURE_COLS."""

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

    run_id_path = Path("run_id.txt")
    if run_id_path.exists():
        RUN_ID = run_id_path.read_text().strip()
        print(f"[startup] Found run_id: {RUN_ID}")
    else:
        print("[startup] run_id.txt not found – health will report 'unknown'.")
        RUN_ID = None

    model_dir = Path("models/model")
    if model_dir.exists():
        try:
            model = mlflow.sklearn.load_model(str(model_dir))
            print(f"[startup] Model loaded from {model_dir}")
        except Exception as e:
            print(f"[startup] Failed to load model: {e}")
            model = None
    else:
        print(f"[startup] Model directory not found: {model_dir}")
        model = None

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
    return {
        "status": "ok" if model is not None else "degraded",
        "run_id": RUN_ID or "unknown",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Check /health.")

    feature_dict = patient.model_dump()
    pred_proba = model.predict_proba([feature_dict])[0, 1]
    return PredictionResponse(
        risk_score=float(pred_proba),
        model_version=RUN_ID or "unknown",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=9696, reload=True)
