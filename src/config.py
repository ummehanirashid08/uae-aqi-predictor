from pathlib import Path
import os
from dotenv import load_dotenv


load_dotenv()


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

FEATURE_STORE_PATH = DATA_DIR / "feature_store.parquet"
MODEL_PATH = MODELS_DIR / "aqi_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

USE_HOPSWORKS = os.getenv("USE_HOPSWORKS", "false").lower() == "true"
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME", "uae_aqi_predictor")

HOPSWORKS_FEATURE_GROUP_NAME = "aqi_features"
HOPSWORKS_FEATURE_GROUP_VERSION = 1

HOPSWORKS_MODEL_NAME = "aqi_forecasting_model"


UAE_LOCATIONS = [
    {
        "city": "Al Ain",
        "latitude": 24.1302,
        "longitude": 55.8023,
    },
    {
        "city": "Abu Dhabi",
        "latitude": 24.4539,
        "longitude": 54.3773,
    },
    {
        "city": "Dubai",
        "latitude": 25.2048,
        "longitude": 55.2708,
    },
    {
        "city": "Sharjah",
        "latitude": 25.3463,
        "longitude": 55.4209,
    },
    {
        "city": "Ajman",
        "latitude": 25.4052,
        "longitude": 55.5136,
    },
    {
        "city": "Ras Al Khaimah",
        "latitude": 25.8007,
        "longitude": 55.9762,
    },
]


FEATURE_COLUMNS = [
    "latitude",
    "longitude",
    "hour",
    "day",
    "month",
    "dayofweek",

    "us_aqi",
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "dust",

    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",

    "aqi_lag_1h",
    "aqi_lag_3h",
    "aqi_lag_6h",
    "aqi_lag_12h",
    "aqi_lag_24h",

    "pm25_lag_24h",
    "pm10_lag_24h",

    "aqi_rolling_mean_3h",
    "aqi_rolling_mean_6h",
    "aqi_rolling_mean_12h",
    "aqi_rolling_mean_24h",

    "pm25_rolling_mean_24h",
    "pm10_rolling_mean_24h",
    "ozone_rolling_mean_24h",
    "wind_speed_rolling_mean_24h",
    "pressure_rolling_mean_24h",

    "aqi_change_rate_1h",
    "aqi_change_rate_3h",
    "aqi_change_rate_24h",

    "future_temp_day1_avg",
    "future_temp_day2_avg",
    "future_temp_day3_avg",

    "future_humidity_day1_avg",
    "future_humidity_day2_avg",
    "future_humidity_day3_avg",

    "future_wind_day1_avg",
    "future_wind_day2_avg",
    "future_wind_day3_avg",

    "future_wind_day1_max",
    "future_wind_day2_max",
    "future_wind_day3_max",

    "future_wind_direction_day1_avg",
    "future_wind_direction_day2_avg",
    "future_wind_direction_day3_avg",

    "future_pressure_day1_avg",
    "future_pressure_day2_avg",
    "future_pressure_day3_avg",

    "future_precip_day1_sum",
    "future_precip_day2_sum",
    "future_precip_day3_sum",
]


TARGET_COLUMNS = {
    "day_1": "target_aqi_day1_avg",
    "day_2": "target_aqi_day2_avg",
    "day_3": "target_aqi_day3_avg",
    "next_72h_avg": "target_aqi_next_72h_avg",
}