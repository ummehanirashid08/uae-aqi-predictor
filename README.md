# UAE AQI Predictor

A machine learning based Air Quality Index prediction system for UAE cities.  
The project collects weather and air pollution data, builds a feature store, trains multiple forecasting models, and serves a Streamlit dashboard for real-time AQI monitoring and 3-day AQI forecasting.

---

## Project Overview

This project predicts Air Quality Index for selected UAE cities using historical pollutant values, weather conditions, rolling trends, lag features, and future weather forecasts.

The system supports:

- Hourly feature pipeline
- Historical data backfill
- Hopsworks Feature Store integration
- Multi-target AQI forecasting
- Daily automated model training
- Model evaluation using RMSE, MAE, and R²
- Streamlit dashboard
- EDA and explainability reports
- GitHub Actions automation

---

## Supported Cities

The dashboard currently supports the following UAE cities:

- Dubai
- Abu Dhabi
- Sharjah
- Ajman
- Al Ain
- Ras Al Khaimah

---

## Architecture

```text
External APIs
    ↓
Feature Pipeline
    ↓
Feature Engineering
    ↓
Hopsworks Feature Store
    ↓
Training Pipeline
    ↓
Model Evaluation
    ↓
Local Model Backup / Model Registry Attempt
    ↓
Streamlit Dashboard
    ↓
EDA and Explainability Reports