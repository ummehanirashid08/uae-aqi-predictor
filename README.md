# Pearls AQI Predictor

A simple end-to-end AQI prediction system using Open-Meteo Air Quality API, local file-based feature store, Scikit-learn model training, GitHub Actions automation, and Streamlit dashboard.

## 1) Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## 2) Backfill data

Dubai example:

```bash
python src/feature_pipeline.py --latitude 25.2048 --longitude 55.2708 --timezone Asia/Dubai --start-date 2026-03-01 --end-date 2026-06-01
```

## 3) Train model

```bash
python src/training_pipeline.py
```

## 4) Run dashboard

```bash
streamlit run src/app.py
```

## 5) Automate with GitHub Actions

Add this repo to GitHub. The workflows in `.github/workflows/` run feature pipeline hourly and training daily.

## Notes

- Current implementation uses a local file-based feature store: `data/feature_store.parquet`.
- Model registry is local: `models/aqi_model.joblib`.
- Later you can replace local storage with Hopsworks, Vertex AI Feature Store, AWS S3, or GCP Cloud Storage.
