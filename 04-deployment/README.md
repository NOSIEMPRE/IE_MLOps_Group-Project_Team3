# 04 – Model Deployment

Deploy the Hospital Readmission Risk model as a REST API using FastAPI + MLflow.

---

## What You Will Do

1. Train the model (`python train.py`) — logs to MLflow, creates `run_id.txt`
2. Start the API (`python app.py`)
3. Explore via Swagger (`http://localhost:9696/docs`)
4. Test the API (`pytest -q test_api.py`)

---

## Folder Structure

```
04-deployment/
├── train.py       # Train & log Pipeline to MLflow, create run_id.txt
├── app.py         # FastAPI service (loads model from MLflow)
├── test_api.py    # Pytest for /health and /predict
├── run_id.txt     # Generated after training
└── README.md
```

---

## 1. Train the Model

Ensure `../data/diabetic_data.csv` exists.

```bash
cd 04-deployment
python train.py
```

Output: PR-AUC, ROC-AUC, and `run_id.txt`. Model is logged to local MLflow (sqlite).

---

## 2. Start the API

```bash
python app.py
```

Service runs at http://localhost:9696

---

## 3. Test via Browser

Open http://localhost:9696/docs

- `POST /predict` → Try it out with the example payload
- Response: `risk_score` (0–1), `model_version`

---

## 4. Automated API Test

```bash
pip install pytest requests
python -m pytest -q test_api.py
```

Expect `2 passed`.

---

## API Endpoints

| Endpoint | Purpose           |
| -------- | ----------------- |
| /        | Welcome/info      |
| /health  | Service + run_id  |
| /predict | Risk prediction   |
| /docs    | Swagger UI        |

---

## Troubleshooting

| Issue                | Fix                     |
| -------------------- | ----------------------- |
| "run_id.txt not found" | Run `python train.py` first |
| Data not found       | Download from UCI, place in `data/diabetic_data.csv` |
| API not responding   | Ensure `python app.py` is running |
