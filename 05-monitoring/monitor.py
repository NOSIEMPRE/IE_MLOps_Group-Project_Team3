"""
Generate Evidently HTML report: data drift + classification performance.

Usage:
    python monitor.py
Requires: data/predictions.csv (from simulate.py)
"""

import pandas as pd
from pathlib import Path

LOG_PATH = Path("data/predictions.csv")
REPORT_PATH = Path("monitoring_report.html")


def main():
    print("\n📊 Starting monitoring report...\n")

    if not LOG_PATH.exists():
        raise FileNotFoundError("❌ No logged predictions found. Run simulate.py first!")

    df = pd.read_csv(LOG_PATH)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["risk_score", "target"])
    print(f"✓ Loaded {len(df)} logged predictions")

    df = df.sort_values("ts" if "ts" in df.columns else df.index)
    midpoint = len(df) // 2
    reference = df.iloc[:midpoint].copy()
    current = df.iloc[midpoint:].copy()

    print(f"Reference: {len(reference)}  |  Current: {len(current)}")

    try:
        from evidently import Dataset, DataDefinition, Report, BinaryClassification
        from evidently.presets import DataDriftPreset, ClassificationPreset
    except ImportError:
        try:
            from evidently import Dataset, DataDefinition, Report
            from evidently.core.datasets import BinaryClassification
            from evidently.presets import DataDriftPreset, ClassificationPreset
        except ImportError as e:
            print("❌ Install evidently: pip install evidently")
            raise e

    num_cols = [c for c in ["time_in_hospital", "num_lab_procedures", "care_intensity"] if c in df.columns]
    cat_cols = [c for c in ["age", "gender", "race"] if c in df.columns]

    data_def = DataDefinition(
        classification=[BinaryClassification(target="target", prediction_labels="predicted_class")],
        numerical_columns=num_cols,
        categorical_columns=cat_cols,
    )
    ref_dataset = Dataset.from_pandas(reference, data_definition=data_def)
    cur_dataset = Dataset.from_pandas(current, data_definition=data_def)

    print("\n🧮 Generating Evidently drift report...")
    report = Report(metrics=[DataDriftPreset(), ClassificationPreset()])
    snapshot = report.run(cur_dataset, ref_dataset)

    snapshot.save_html(str(REPORT_PATH))
    print(f"✅ Report saved: {REPORT_PATH.resolve()}")
    print("Open it in your browser to explore drift metrics.\n")


if __name__ == "__main__":
    main()
