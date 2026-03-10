# 05 – Monitoring: Train → Serve → Simulate → Drift Detection

Complete MLOps loop: train → serve → simulate production traffic → monitor drift.

---

## Folder Structure

```
05-monitoring/
├── train.py       # Train & log to MLflow, create run_id.txt
├── app.py         # FastAPI service
├── simulate.py    # Simulates predictions → data/predictions.csv
├── monitor.py     # Evidently drift report → monitoring_report.html
├── test_api.py    # API tests
└── README.md
```

---

## 1. Train the Model

```bash
cd 05-monitoring
python train.py
```

Ensure `../data/diabetic_data.csv` exists.

---

## 2. Start the API

```bash
python app.py
```

Service: http://localhost:9696

---

## 3. Simulate Production Traffic

In a **new terminal** (API must be running):

```bash
cd 05-monitoring
python simulate.py
```

- Calls `/predict` ~100 times with real hospital data
- Logs predictions + ground truth to `data/predictions.csv`

---

## 4. Generate Monitoring Report

```bash
pip install evidently
python monitor.py
```

- Loads `data/predictions.csv`
- Splits into reference (older) vs current (newer)
- Generates Evidently report: **Data Drift** + **Classification Performance**
- Saves `monitoring_report.html`

---

## 5. View Results

- Open `monitoring_report.html` in your browser
- Data drift: input feature distributions
- Classification: confusion matrix, precision, recall, F1, ROC AUC

---

## Quick Commands

| Action | Command |
|--------|---------|
| Train | `python train.py` |
| Run API | `python app.py` |
| Simulate | `python simulate.py` |
| Monitor | `python monitor.py` |
| Test API | `python -m pytest -q test_api.py` |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| No predictions logged | Run `app.py` before `simulate.py` |
| `FileNotFoundError` in monitor | Run `simulate.py` first |
| `ModuleNotFoundError: evidently` | `pip install evidently` |
