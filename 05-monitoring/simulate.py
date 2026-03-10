"""
Simulate production requests to the Hospital Readmission Risk API and log predictions.

Usage:
    python simulate.py
Requires: API running (python app.py)
"""

import time
import requests
import pandas as pd
from pathlib import Path

API_URL = "http://localhost:9696/predict"
LOG_PATH = Path("data/predictions.csv")
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "diabetic_data.csv"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_emergency", "number_inpatient", "number_outpatient", "number_diagnoses",
    "care_intensity", "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "age", "gender", "race", "change", "diabetesMed", "medication_changed",
    "A1Cresult", "max_glu_serum",
]


def load_data(n_rows: int = 100) -> pd.DataFrame:
    """Load and preprocess hospital data for simulation."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data not found at {DATA_PATH}. Run from project root.")
    df = pd.read_csv(DATA_PATH)
    df["target"] = df["readmitted"].isin(["30", "<30"]).astype(int)
    if "weight" in df.columns:
        df = df.drop(columns=["weight"])
    for col in ["age", "gender", "change", "diabetesMed"]:
        df[col] = df[col].fillna("Unknown").astype(str)
    for col in ["A1Cresult", "max_glu_serum"]:
        if col not in df.columns:
            df[col] = "not_tested"
        else:
            df[col] = df[col].fillna("None").replace("None", "not_tested").astype(str)
    if "race" in df.columns:
        df["race"] = df["race"].fillna("Unknown").replace("?", "Unknown").astype(str)
    for col in ["number_emergency", "number_inpatient", "number_outpatient"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0).astype(int)
    df["care_intensity"] = df["number_emergency"] + df["number_inpatient"] + df["number_outpatient"]
    df["medication_changed"] = (df["change"] == "Ch").astype(int)
    for col in ["num_lab_procedures", "num_procedures", "num_medications", "number_diagnoses"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0).astype(int)

    cols = [c for c in FEATURE_COLS if c in df.columns]
    df = df[cols + ["target"]].dropna(subset=cols)
    df = df.sample(n=min(n_rows, len(df)), random_state=42).reset_index(drop=True)
    print(f"✓ Loaded {len(df)} rows for simulation")
    return df


def simulate_requests(df: pd.DataFrame, sleep_s: float = 0.05) -> pd.DataFrame:
    """Send each row to /predict and collect results."""
    rows = []
    for i, row in df.iterrows():
        payload = {}
        int_cols = {"time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
                    "number_emergency", "number_inpatient", "number_outpatient", "number_diagnoses",
                    "care_intensity", "medication_changed", "admission_type_id", "discharge_disposition_id", "admission_source_id"}
        for c in FEATURE_COLS:
            if c not in row.index:
                continue
            v = row[c]
            if pd.isna(v):
                payload[c] = 0 if c in int_cols else "Unknown"
            elif c in int_cols:
                payload[c] = int(v)
            else:
                payload[c] = str(v)

        try:
            resp = requests.post(API_URL, json=payload, timeout=5)
            resp.raise_for_status()
            risk_score = resp.json()["risk_score"]
            predicted_class = 1 if risk_score >= 0.5 else 0

            log_row = {
                "ts": pd.Timestamp.utcnow().isoformat(),
                "target": int(row["target"]),
                "risk_score": risk_score,
                "predicted_class": predicted_class,
            }
            for c in ["time_in_hospital", "num_lab_procedures", "care_intensity", "age", "gender", "race"]:
                if c in row.index:
                    log_row[c] = row[c]
            rows.append(log_row)

        except Exception as e:
            print(f"⚠️  Request failed: {e}")

        if (i + 1) % 20 == 0:
            print(f"   Progress: {i + 1}/{len(df)}")
        time.sleep(sleep_s)

    return pd.DataFrame(rows)


def main():
    print("\n🏥 Starting simulation...\n")
    df = load_data(n_rows=100)
    out = simulate_requests(df)

    if out.empty:
        print("❌ No predictions recorded. Make sure app.py is running.")
        return

    if LOG_PATH.exists():
        prev = pd.read_csv(LOG_PATH)
        out = pd.concat([prev, out], ignore_index=True)

    out.to_csv(LOG_PATH, index=False)
    print(f"✅ Wrote {len(out)} total rows to {LOG_PATH}\n")


if __name__ == "__main__":
    main()
