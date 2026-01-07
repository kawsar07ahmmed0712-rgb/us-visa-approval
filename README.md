# US Visa Approval Prediction — End-to-End ML Pipeline (Local)



An end-to-end machine learning project that predicts **US visa case outcome** (e.g., `CERTIFIED` vs `DENIED`) using a structured dataset.  
This repository includes a full **local** ML pipeline, saved artifacts, a Flask inference API, and a React (Vite) frontend UI.

---

## Key Features

- ✅ Full ML pipeline (local): ingestion → validation → transformation → training → evaluation  
- ✅ Feature engineering (e.g., `company_age` from `yr_of_estab`)  
- ✅ Robust preprocessing (categorical encoding + numeric transforms)  
- ✅ Model training + evaluation + threshold-based decisioning  
- ✅ Saved artifacts: `preprocessor.pkl`, `model.pkl`, `metrics.json`, `evaluation_report.json`  
- ✅ Flask API: `/health`, `/meta`, `/predict`  
- ✅ React UI: form inputs, prediction result, payload viewer, history, meta panel  

---

## Tech Stack

### Backend / ML
- Python 3.10+ (recommended)
- pandas, numpy
- scikit-learn
- joblib
- Flask, Flask-CORS

### Frontend
- React + Vite
- JavaScript

---

## Project Structure

```text
us-visa-approval/
├─ artifacts/
│  ├─ data/
│  │  ├─ raw.csv
│  │  ├─ train.csv
│  │  └─ test.csv
│  ├─ transformation/
│  │  └─ preprocessor.pkl
│  ├─ model_trainer/
│  │  ├─ model.pkl
│  │  └─ metrics.json
│  ├─ model_evaluation/
│  │  └─ evaluation_report.json
│  └─ validation/
│     ├─ schema_inferred.yaml
│     ├─ missing_report.csv
│     └─ validation_report.json
│
├─ backend/
│  └─ app.py
│
├─ frontend/
│  ├─ src/
│  │  └─ App.jsx
│  ├─ package.json
│  └─ ...
│
├─ notebook/
│  └─ visadataset.csv
│
├─ src/
│  └─ ... (pipeline components)
│
├─ main.py
├─ predict.py
└─ README.md
````

---

## Dataset

This project expects a CSV dataset (example: `notebook/visadataset.csv`) with a target column similar to:

* `case_status` (values like `Certified`, `Denied`)

If your dataset path or target column differs, update ingestion/config accordingly.

---

## Prerequisites

* Python 3.10+ recommended
* Node.js 18+ (LTS recommended)
* npm (included with Node)

---

## Local Setup

### 1) Create & activate a Python environment

Using Conda:

```bash
conda create -n visa python=3.10 -y
conda activate visa
```

### 2) Install Python dependencies

If you have a `requirements.txt`:

```bash
pip install -r requirements.txt
```

If you do not have `requirements.txt`, install core packages:

```bash
pip install pandas numpy scikit-learn joblib flask flask-cors
```

### 3) Install frontend dependencies

```bash
cd frontend
npm install
```

---

## Train the Model (Run Full Pipeline)

From the project root:

```bash
python main.py
```

This generates artifacts under `artifacts/`, including:

* `artifacts/transformation/preprocessor.pkl`
* `artifacts/model_trainer/model.pkl`
* `artifacts/model_trainer/metrics.json`
* `artifacts/model_evaluation/evaluation_report.json`

---

## Run Backend (Flask API)

From the project root:

```bash
python backend/app.py
```

Backend runs at:

* `http://127.0.0.1:5000`

---

## API Documentation

### 1) Health Check

```http
GET /health
```

Example response:

```json
{
  "ok": true,
  "threshold": 0.55,
  "label_pos": "CERTIFIED",
  "label_neg": "DENIED"
}
```

### 2) Meta (Dropdown Options + Numeric Ranges)

```http
GET /meta
```

* Reads from `artifacts/data/train.csv` (if available)
* Returns categorical values for dropdowns and numeric ranges for hints

### 3) Predict

```http
POST /predict
Content-Type: application/json
```

Example request:

```json
{
  "continent": "Asia",
  "region_of_employment": "Northeast",
  "unit_of_wage": "Year",
  "education_of_employee": "Bachelor's",
  "has_job_experience": "Y",
  "requires_job_training": "N",
  "full_time_position": "Y",
  "no_of_employees": 50,
  "prevailing_wage": 70000,
  "yr_of_estab": 2005
}
```

Example response:

```json
{
  "ok": true,
  "label": "CERTIFIED",
  "pred": 1,
  "score_pos_class": 0.9536,
  "threshold": 0.55
}
```

---

## Run Frontend (React + Vite)

From the project root:

```bash
cd frontend
npm run dev
```

Frontend runs at:

* `http://localhost:5173`

---

## Configure Frontend API URL (Optional)

Create a file at:

```text
frontend/.env
```

Add:

```env
VITE_API_URL=http://127.0.0.1:5000
```

---

## CLI Prediction (Optional)

If your repo includes `predict.py`:

```bash
python predict.py --from-test 0
python predict.py --from-test 10
```

---

## Artifacts & Reproducibility

The backend loads:

* `artifacts/transformation/preprocessor.pkl`
* `artifacts/model_trainer/model.pkl`
* `threshold` from `artifacts/model_evaluation/evaluation_report.json` (if present)

If artifacts are missing, regenerate them:

```bash
python main.py
```

---

## Troubleshooting

### `npm run dev` → `package.json` not found

Run inside the `frontend/` directory:

```bash
cd frontend
npm run dev
```

### Backend `/health` returns `"ok": false`

Common reasons:

* Missing artifacts (`preprocessor.pkl` or `model.pkl`) → run `python main.py`
* Python environment mismatch (packages differ from training environment)

### `ModuleNotFoundError: No module named 'src'` while loading artifacts

This happens if the pickled artifacts reference `src.*`.
Ensure:

* `src/__init__.py` exists
* `backend/app.py` adds the project root to `sys.path` before `joblib.load(...)`



