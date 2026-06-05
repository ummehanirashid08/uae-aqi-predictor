import os
import json
import time
import joblib
import requests
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.inspection import permutation_importance

from config import FEATURE_COLUMNS, FEATURE_STORE_PATH, MODEL_PATH, METRICS_PATH
from hopsworks_store import load_features_from_hopsworks


os.environ["LOKY_MAX_CPU_COUNT"] = "4"


CITY_MAP = {
    "24.1302,55.8023": "Al Ain",
    "24.4539,54.3773": "Abu Dhabi",
    "25.2048,55.2708": "Dubai",
    "25.3463,55.4209": "Sharjah",
    "25.4052,55.5136": "Ajman",
    "25.8007,55.9762": "Ras Al Khaimah",
}


DAY_OPTIONS = {
    "Day 1 Forecast": "day_1",
    "Day 2 Forecast": "day_2",
    "Day 3 Forecast": "day_3",
    "3-Day Average": "next_72h_avg",
}


POLLUTANT_COLUMNS = [
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "dust",
]


POLLUTANT_LABELS = {
    "pm2_5": "Particulate Matter",
    "pm10": "Particulate Matter",
    "carbon_monoxide": "Carbon Monoxide",
    "nitrogen_dioxide": "Nitrogen Dioxide",
    "sulphur_dioxide": "Sulphur Dioxide",
    "ozone": "Ozone",
    "dust": "Dust",
}


POLLUTANT_SHORT = {
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "carbon_monoxide": "CO",
    "nitrogen_dioxide": "NO₂",
    "sulphur_dioxide": "SO₂",
    "ozone": "O₃",
    "dust": "Dust",
}


POLLUTANT_UNITS = {
    "pm2_5": "µg/m³",
    "pm10": "µg/m³",
    "carbon_monoxide": "µg/m³",
    "nitrogen_dioxide": "µg/m³",
    "sulphur_dioxide": "µg/m³",
    "ozone": "µg/m³",
    "dust": "µg/m³",
}


POLLUTANT_ICONS = {
    "pm2_5": "🌫️",
    "pm10": "💨",
    "carbon_monoxide": "☁️",
    "nitrogen_dioxide": "🏭",
    "sulphur_dioxide": "🌧️",
    "ozone": "⭕",
    "dust": "🏜️",
}


THEME = {
    "card_bg": "#fffdf2",
    "yellow_border": "#f3e8a2",
    "yellow": "#facc15",
    "blue": "#2563eb",
    "cyan": "#0ea5e9",
    "text": "#0f172a",
    "muted": "#64748b",
}


def normalize_timestamp(ts):
    parsed_ts = pd.to_datetime(ts, errors="coerce")

    if pd.isna(parsed_ts):
        return None

    return parsed_ts


def is_feature_store_old(latest_time, days=3):
    latest_ts = normalize_timestamp(latest_time)

    if latest_ts is None:
        return False

    if latest_ts.tzinfo is not None:
        now_ts = pd.Timestamp.now(tz=latest_ts.tzinfo)
    else:
        now_ts = pd.Timestamp.now()

    return latest_ts < (now_ts - pd.Timedelta(days=days))


def clean_model_name(model_name: str):
    if not model_name:
        return "Unknown Model"

    name_map = {
        "hist_gradient_boosting": "Histogram Gradient Boosting",
        "gradient_boosting": "Gradient Boosting",
        "random_forest": "Random Forest",
        "extra_trees": "Extra Trees",
        "xgboost": "XGBoost",
        "ridge": "Ridge Regression",
        "lasso": "Lasso Regression",
        "elastic_net": "Elastic Net",
        "baseline_current_aqi": "Baseline AQI",
    }

    return name_map.get(model_name, str(model_name).replace("_", " ").title())


def clean_target_name(target_key: str):
    target_map = {
        "day_1": "Next 24 Hours",
        "day_2": "24–48 Hours Ahead",
        "day_3": "48–72 Hours Ahead",
        "next_72h_avg": "3-Day Average",
    }

    return target_map.get(target_key, target_key)


def get_accuracy_label(r2_value):
    if r2_value is None:
        return "Not Available", "Model score is not available.", "#64748b", "⚪"

    if r2_value >= 0.80:
        return "Strong", "The model explains most AQI changes well.", "#22c55e", "✅"

    if r2_value >= 0.60:
        return "Good", "The model captures the main AQI patterns.", "#2563eb", "👍"

    if r2_value >= 0.40:
        return "Moderate", "The model gives a useful estimate but may vary.", "#eab308", "⚠️"

    return "Needs Improvement", "The model has limited accuracy for this forecast.", "#ef4444", "🔎"


def get_aqi_category(aqi: float):
    if aqi <= 50:
        return "Good", "Air quality is satisfactory.", "🟢", "#22c55e"
    elif aqi <= 100:
        return "Moderate", "Air quality is acceptable for most people.", "🟡", "#eab308"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "Sensitive people should reduce prolonged outdoor activity.", "🟠", "#f97316"
    elif aqi <= 200:
        return "Unhealthy", "Everyone may begin to experience health effects. Outdoor activity should be reduced.", "🔴", "#ef4444"
    elif aqi <= 300:
        return "Very Unhealthy", "Health alert. Avoid outdoor activity where possible.", "🟣", "#8b5cf6"
    else:
        return "Hazardous", "Emergency conditions. Stay indoors and avoid outdoor exposure.", "⚫", "#111827"


def get_pollutant_color(pollutant: str, value: float):
    if pollutant == "pm2_5":
        if value <= 12:
            return "#22c55e"
        elif value <= 35:
            return "#eab308"
        elif value <= 55:
            return "#f97316"
        return "#ef4444"

    if pollutant == "pm10":
        if value <= 54:
            return "#22c55e"
        elif value <= 154:
            return "#eab308"
        elif value <= 254:
            return "#f97316"
        return "#ef4444"

    if pollutant == "carbon_monoxide":
        if value <= 1000:
            return "#22c55e"
        elif value <= 4000:
            return "#eab308"
        elif value <= 10000:
            return "#f97316"
        return "#ef4444"

    if pollutant == "nitrogen_dioxide":
        if value <= 40:
            return "#22c55e"
        elif value <= 100:
            return "#eab308"
        elif value <= 200:
            return "#f97316"
        return "#ef4444"

    if pollutant == "sulphur_dioxide":
        if value <= 20:
            return "#22c55e"
        elif value <= 80:
            return "#eab308"
        elif value <= 250:
            return "#f97316"
        return "#ef4444"

    if pollutant == "ozone":
        if value <= 100:
            return "#22c55e"
        elif value <= 160:
            return "#eab308"
        elif value <= 240:
            return "#f97316"
        return "#ef4444"

    if pollutant == "dust":
        if value <= 50:
            return "#22c55e"
        elif value <= 150:
            return "#eab308"
        elif value <= 300:
            return "#f97316"
        return "#ef4444"

    return "#2563eb"


def get_health_recommendations(aqi: float):
    if aqi <= 50:
        return [
            "Outdoor activity is generally safe.",
            "Keep monitoring AQI if conditions change.",
            "No special precautions are needed.",
        ]

    if aqi <= 100:
        return [
            "Most people can continue normal outdoor activities.",
            "Sensitive people should monitor symptoms.",
            "Keep windows closed if air quality worsens.",
        ]

    if aqi <= 150:
        return [
            "Sensitive groups should reduce prolonged outdoor exertion.",
            "People with asthma should carry medication.",
            "Consider wearing a mask outdoors.",
        ]

    if aqi <= 200:
        return [
            "Limit outdoor exercise.",
            "Children and elderly people should stay indoors.",
            "Keep doors and windows closed where possible.",
        ]

    if aqi <= 300:
        return [
            "Avoid outdoor activity unless necessary.",
            "Use air purification indoors if available.",
            "Sensitive groups should avoid exposure completely.",
        ]

    return [
        "Stay indoors.",
        "Use air purifiers and keep windows closed.",
        "Follow official health and environmental alerts.",
    ]


@st.cache_data(ttl=15 * 60, show_spinner=False)
def fetch_live_current_aqi(latitude: float, longitude: float, timezone: str = "Asia/Dubai"):
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    params = {
        "latitude": round(float(latitude), 4),
        "longitude": round(float(longitude), 4),
        "timezone": timezone,
        "current": ",".join([
            "us_aqi",
            "pm10",
            "pm2_5",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
            "dust",
        ]),
    }

    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=30)

            if response.status_code in [429, 502, 503, 504]:
                time.sleep(2 + attempt * 2)
                continue

            response.raise_for_status()
            data = response.json()
            return data.get("current")

        except Exception as error:
            print("Live AQI API error:", error)
            time.sleep(2 + attempt * 2)

    return None


@st.cache_data(ttl=15 * 60, show_spinner=False)
def fetch_live_current_weather(latitude: float, longitude: float, timezone: str = "Asia/Dubai"):
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": round(float(latitude), 4),
        "longitude": round(float(longitude), 4),
        "timezone": timezone,
        "current": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "precipitation",
            "weather_code",
        ]),
    }

    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=30)

            if response.status_code in [429, 502, 503, 504]:
                time.sleep(2 + attempt * 2)
                continue

            response.raise_for_status()
            data = response.json()
            return data.get("current")

        except Exception as error:
            print("Live weather API current error:", error)
            time.sleep(2 + attempt * 2)

    return None


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_future_weather_forecast(latitude: float, longitude: float, timezone: str = "Asia/Dubai"):
    url = "https://api.open-meteo.com/v1/forecast"

    variable_sets = [
        [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
        ],
        [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "pressure_msl",
            "wind_speed_10m",
            "wind_direction_10m",
        ],
        [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_direction_10m",
        ],
    ]

    for variables in variable_sets:
        params = {
            "latitude": round(float(latitude), 4),
            "longitude": round(float(longitude), 4),
            "timezone": timezone,
            "forecast_days": 4,
            "hourly": ",".join(variables),
        }

        for attempt in range(3):
            try:
                response = requests.get(url, params=params, timeout=30)

                if response.status_code in [429, 502, 503, 504]:
                    time.sleep(2 + attempt * 2)
                    continue

                response.raise_for_status()
                data = response.json()

                if "hourly" not in data:
                    continue

                forecast_df = pd.DataFrame(data["hourly"])

                if forecast_df.empty:
                    continue

                forecast_df["time"] = pd.to_datetime(forecast_df["time"])

                if "surface_pressure" not in forecast_df.columns and "pressure_msl" in forecast_df.columns:
                    forecast_df["surface_pressure"] = forecast_df["pressure_msl"]

                if "surface_pressure" not in forecast_df.columns:
                    forecast_df["surface_pressure"] = forecast_df["temperature_2m"] * 0 + 1008

                return forecast_df

            except Exception as error:
                print("Future weather API error:", error)
                time.sleep(2 + attempt * 2)

    return None


def get_weather_icon(weather_code):
    if weather_code is None:
        return "🌤️"

    try:
        code = int(weather_code)
    except Exception:
        return "🌤️"

    if code == 0:
        return "☀️"
    if code in [1, 2, 3]:
        return "🌤️"
    if code in [45, 48]:
        return "🌫️"
    if code in [51, 53, 55, 61, 63, 65, 80, 81, 82]:
        return "🌧️"
    if code in [95, 96, 99]:
        return "⛈️"

    return "🌤️"


def safe_mean(series):
    if series is None or len(series) == 0:
        return None

    value = series.mean()

    if pd.isna(value):
        return None

    return float(value)


def safe_sum(series):
    if series is None or len(series) == 0:
        return None

    value = series.sum()

    if pd.isna(value):
        return None

    return float(value)


def safe_max(series):
    if series is None or len(series) == 0:
        return None

    value = series.max()

    if pd.isna(value):
        return None

    return float(value)


def update_latest_row_with_future_weather(latest_row, future_weather_df):
    updated_row = latest_row.copy()

    if future_weather_df is None or future_weather_df.empty:
        return updated_row, False

    future_weather_df = future_weather_df.sort_values("time").reset_index(drop=True)

    if len(future_weather_df) < 72:
        return updated_row, False

    windows = {
        "day1": future_weather_df.iloc[0:24],
        "day2": future_weather_df.iloc[24:48],
        "day3": future_weather_df.iloc[48:72],
    }

    mapping = {
        "temperature_2m": "temp",
        "relative_humidity_2m": "humidity",
        "wind_speed_10m": "wind",
        "wind_direction_10m": "wind_direction",
        "surface_pressure": "pressure",
    }

    applied_count = 0

    for day_name, window_df in windows.items():
        for source_col, short_name in mapping.items():
            feature_name = f"future_{short_name}_{day_name}_avg"

            if source_col in window_df.columns:
                value = safe_mean(window_df[source_col])

                if value is not None and feature_name in updated_row.index:
                    updated_row[feature_name] = value
                    applied_count += 1

        precip_feature = f"future_precip_{day_name}_sum"

        if "precipitation" in window_df.columns and precip_feature in updated_row.index:
            value = safe_sum(window_df["precipitation"])

            if value is not None:
                updated_row[precip_feature] = value
                applied_count += 1

        wind_max_feature = f"future_wind_{day_name}_max"

        if "wind_speed_10m" in window_df.columns and wind_max_feature in updated_row.index:
            value = safe_max(window_df["wind_speed_10m"])

            if value is not None:
                updated_row[wind_max_feature] = value
                applied_count += 1

    return updated_row, applied_count > 0


def get_future_weather_summary(future_weather_df):
    if future_weather_df is None or future_weather_df.empty or len(future_weather_df) < 72:
        return pd.DataFrame()

    future_weather_df = future_weather_df.sort_values("time").reset_index(drop=True)

    windows = {
        "Day 1": future_weather_df.iloc[0:24],
        "Day 2": future_weather_df.iloc[24:48],
        "Day 3": future_weather_df.iloc[48:72],
    }

    rows = []

    for day_name, window_df in windows.items():
        rows.append({
            "Day": day_name,
            "Avg Temp": round(safe_mean(window_df["temperature_2m"]), 1) if "temperature_2m" in window_df else None,
            "Avg Humidity": round(safe_mean(window_df["relative_humidity_2m"]), 1) if "relative_humidity_2m" in window_df else None,
            "Avg Wind": round(safe_mean(window_df["wind_speed_10m"]), 1) if "wind_speed_10m" in window_df else None,
            "Max Wind": round(safe_max(window_df["wind_speed_10m"]), 1) if "wind_speed_10m" in window_df else None,
            "Rain Total": round(safe_sum(window_df["precipitation"]), 2) if "precipitation" in window_df else None,
        })

    return pd.DataFrame(rows)


@st.cache_data(ttl=15 * 60, show_spinner=False)
def load_feature_store():
    cloud_df, cloud_source = load_features_from_hopsworks()

    if cloud_df is not None and not cloud_df.empty:
        cloud_df["time"] = pd.to_datetime(cloud_df["time"], errors="coerce")
        cloud_df = cloud_df.dropna(subset=["time"])
        cloud_df = cloud_df.sort_values("time").reset_index(drop=True)
        return cloud_df, cloud_source

    if not FEATURE_STORE_PATH.exists():
        st.error("Feature store not found in Hopsworks or local parquet. Please run feature_pipeline.py first.")
        st.stop()

    local_df = pd.read_parquet(FEATURE_STORE_PATH)
    local_df["time"] = pd.to_datetime(local_df["time"], errors="coerce")
    local_df = local_df.dropna(subset=["time"])
    local_df = local_df.sort_values("time").reset_index(drop=True)

    return local_df, "Local Parquet fallback"


def load_model_bundle():
    if not MODEL_PATH.exists():
        st.error("Model not found. Please run training_pipeline.py first.")
        st.stop()

    bundle = joblib.load(MODEL_PATH)

    if not isinstance(bundle, dict) or "models" not in bundle:
        st.error("Old model format found. Please run training_pipeline.py again.")
        st.stop()

    return bundle


def load_metrics():
    if not METRICS_PATH.exists():
        return None

    with open(METRICS_PATH, "r") as file:
        return json.load(file)


def create_city_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["location_lookup_key"] = (
        df["latitude"].round(4).astype(str)
        + ","
        + df["longitude"].round(4).astype(str)
    )

    mapped_city = df["location_lookup_key"].map(CITY_MAP)

    if "city" in df.columns:
        df["city"] = df["city"].astype(str)
        df["city"] = df["city"].replace(["None", "none", "nan", "NaN", "NULL", "null", ""], pd.NA)
        df["city"] = df["city"].fillna(mapped_city)
    else:
        df["city"] = mapped_city

    df["city"] = df["city"].fillna(df["location_lookup_key"])
    df["city"] = df["city"].astype(str)

    df = df[df["city"].str.lower() != "none"].copy()

    return df


def make_pollutant_table(latest_row, live_current=None):
    rows = []

    for col in POLLUTANT_COLUMNS:
        value = None

        if live_current and col in live_current and live_current[col] is not None:
            value = live_current[col]
        elif col in latest_row.index:
            value = latest_row[col]

        if value is not None and not pd.isna(value):
            rows.append({
                "pollutant_key": col,
                "Pollutant": POLLUTANT_LABELS.get(col, col),
                "Short": POLLUTANT_SHORT.get(col, col),
                "Value": round(float(value), 2),
                "Unit": POLLUTANT_UNITS.get(col, ""),
                "Icon": POLLUTANT_ICONS.get(col, "🌫️"),
                "Color": get_pollutant_color(col, float(value)),
            })

    return pd.DataFrame(rows)


def prepare_results_table(metrics, target_key):
    if not metrics:
        return pd.DataFrame()

    target_metrics = metrics.get("targets", {}).get(target_key, {})
    all_results = target_metrics.get("all_model_results", {})

    rows = []

    for model_name, result in all_results.items():
        if isinstance(result, dict) and "rmse" in result:
            rows.append({
                "Model": clean_model_name(model_name),
                "Model Key": model_name,
                "RMSE": result.get("rmse"),
                "MAE": result.get("mae"),
                "R²": result.get("r2"),
            })

    results_df = pd.DataFrame(rows)

    if not results_df.empty:
        results_df = results_df.sort_values("R²", ascending=False)

    return results_df


def get_selected_model_metrics(metrics, target_key, model_name):
    if not metrics:
        return None

    target_metrics = metrics.get("targets", {}).get(target_key, {})
    all_results = target_metrics.get("all_model_results", {})

    return all_results.get(model_name)


def get_target_column(metrics, target_key):
    if metrics:
        return metrics.get("targets", {}).get(target_key, {}).get("target_column")

    fallback = {
        "day_1": "target_aqi_day1_avg",
        "day_2": "target_aqi_day2_avg",
        "day_3": "target_aqi_day3_avg",
        "next_72h_avg": "target_aqi_next_72h_avg",
    }

    return fallback.get(target_key)


def predict_for_target(model_bundle, target_key, latest_row):
    X_latest = pd.DataFrame([latest_row[FEATURE_COLUMNS]])
    selected_model_info = model_bundle["models"][target_key]
    selected_model = selected_model_info["model"]
    selected_model_name = selected_model_info["model_name"]
    prediction = float(selected_model.predict(X_latest)[0])

    return prediction, selected_model_name


def get_trend_status(current_aqi, predicted_aqi):
    difference = predicted_aqi - current_aqi

    if difference > 10:
        return "Worsening", f"+{round(difference, 2)}", "▲", "#ef4444"
    elif difference < -10:
        return "Improving", f"{round(difference, 2)}", "▼", "#22c55e"
    else:
        return "Stable", f"{round(difference, 2)}", "●", "#eab308"


def get_feature_importance(model_bundle, target_key, clean_df, metrics):
    try:
        model_info = model_bundle["models"][target_key]
        model = model_info["model"]

        estimator = model.named_steps["model"] if hasattr(model, "named_steps") and "model" in model.named_steps else model

        if hasattr(estimator, "feature_importances_"):
            values = estimator.feature_importances_

            if hasattr(values, "ravel"):
                values = values.ravel()

            importance_df = pd.DataFrame({
                "Feature": FEATURE_COLUMNS,
                "Importance": values,
            })

            return importance_df.sort_values("Importance", ascending=False).head(15)

        if hasattr(estimator, "coef_"):
            values = abs(estimator.coef_)

            if hasattr(values, "ravel"):
                values = values.ravel()

            importance_df = pd.DataFrame({
                "Feature": FEATURE_COLUMNS,
                "Importance": values,
            })

            return importance_df.sort_values("Importance", ascending=False).head(15)

        target_column = get_target_column(metrics, target_key)

        if target_column and target_column in clean_df.columns:
            sample_df = clean_df.dropna(subset=FEATURE_COLUMNS + [target_column]).tail(500).copy()

            if len(sample_df) >= 50:
                X_sample = sample_df[FEATURE_COLUMNS]
                y_sample = sample_df[target_column]

                result = permutation_importance(
                    model,
                    X_sample,
                    y_sample,
                    n_repeats=3,
                    random_state=42,
                    scoring="neg_mean_absolute_error",
                )

                importance_df = pd.DataFrame({
                    "Feature": FEATURE_COLUMNS,
                    "Importance": result.importances_mean,
                })

                importance_df["Importance"] = importance_df["Importance"].abs()

                return importance_df.sort_values("Importance", ascending=False).head(15)

        return pd.DataFrame()

    except Exception as error:
        print("Feature importance error:", error)
        return pd.DataFrame()


def get_driver_label(feature_name):
    feature_name = str(feature_name)

    if feature_name in ["us_aqi", "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_24h"]:
        return "Recent AQI level"

    if "pm2_5" in feature_name or "pm25" in feature_name:
        return "Fine particle pollution"

    if "pm10" in feature_name:
        return "Dust and larger particles"

    if "rolling" in feature_name or "change_rate" in feature_name:
        return "Recent pollution trend"

    if "ozone" in feature_name:
        return "Ozone level"

    if "nitrogen" in feature_name or "sulphur" in feature_name or "carbon" in feature_name:
        return "Gas pollutant level"

    if "future_temp" in feature_name or "temperature" in feature_name:
        return "Temperature condition"

    if "future_wind" in feature_name or "wind" in feature_name:
        return "Wind movement"

    if "humidity" in feature_name:
        return "Humidity condition"

    if "pressure" in feature_name:
        return "Air pressure"

    if "hour" in feature_name or "month" in feature_name or "dayofweek" in feature_name:
        return "Time pattern"

    return "Environmental factor"


def build_simple_driver_cards(importance_df, current_aqi, predicted_aqi, pollutant_df, future_weather_applied):
    top_features = importance_df.head(10)["Feature"].tolist() if not importance_df.empty else []
    labels = [get_driver_label(feature) for feature in top_features]

    has_aqi = any(label == "Recent AQI level" for label in labels)
    has_pm25 = any(label == "Fine particle pollution" for label in labels)
    has_pm10 = any(label == "Dust and larger particles" for label in labels)
    has_trend = any(label == "Recent pollution trend" for label in labels)
    has_weather = any(label in ["Temperature condition", "Wind movement", "Humidity condition", "Air pressure"] for label in labels)

    trend_status, _, trend_icon, trend_color = get_trend_status(current_aqi, predicted_aqi)

    cards = []

    if has_aqi:
        cards.append({
            "icon": "📍",
            "title": "Current air quality is important",
            "text": "The latest AQI level is one of the strongest signals. When current AQI is already high, the model expects pollution risk to remain elevated.",
            "badge": f"Live AQI: {round(current_aqi, 1)}",
            "color": "#2563eb",
        })

    if has_pm25:
        pm25_value = "-"
        if not pollutant_df.empty:
            pm25_match = pollutant_df[pollutant_df["Short"] == "PM2.5"]
            if not pm25_match.empty:
                pm25_value = pm25_match["Value"].iloc[0]

        cards.append({
            "icon": "🌫️",
            "title": "Fine particles are affecting the forecast",
            "text": "PM2.5 particles are very small and can stay in the air longer. Higher PM2.5 usually increases health risk and forecasted AQI.",
            "badge": f"PM2.5: {pm25_value}",
            "color": "#ef4444",
        })

    if has_pm10:
        pm10_value = "-"
        if not pollutant_df.empty:
            pm10_match = pollutant_df[pollutant_df["Short"] == "PM10"]
            if not pm10_match.empty:
                pm10_value = pm10_match["Value"].iloc[0]

        cards.append({
            "icon": "💨",
            "title": "Dust and larger particles matter",
            "text": "PM10 shows larger particles such as dust. In UAE cities, dust can strongly influence air quality forecasts.",
            "badge": f"PM10: {pm10_value}",
            "color": "#ca8a04",
        })

    if has_trend:
        cards.append({
            "icon": "📈",
            "title": "Recent trend is guiding the model",
            "text": "The model checks whether AQI has been rising, falling, or staying stable over recent hours. This helps it estimate the next day risk.",
            "badge": f"Trend: {trend_icon} {trend_status}",
            "color": trend_color,
        })

    if has_weather:
        cards.append({
            "icon": "🌤️",
            "title": "Weather affects pollution movement",
            "text": "Wind, temperature, humidity, and pressure affect how pollution spreads or stays trapped near the city.",
            "badge": "Live forecast weather used" if future_weather_applied else "Saved weather used",
            "color": "#0ea5e9",
        })

    if len(cards) == 0:
        cards.append({
            "icon": "🤖",
            "title": "Model used multiple signals",
            "text": "The model combined AQI history, pollutant values, city location, time patterns, and weather features to create this forecast.",
            "badge": "ML forecast",
            "color": "#7c3aed",
        })

    return cards[:4]


def style_chart(fig, height=430):
    fig.update_layout(
        height=height,
        plot_bgcolor=THEME["card_bg"],
        paper_bgcolor=THEME["card_bg"],
        font=dict(
            family="Inter, Arial, sans-serif",
            color=THEME["text"],
            size=13,
        ),
        title_font=dict(
            size=22,
            color=THEME["text"],
            family="Inter, Arial, sans-serif",
        ),
        margin=dict(l=24, r=24, t=70, b=40),
        xaxis=dict(
            gridcolor="rgba(148, 163, 184, 0.22)",
            zerolinecolor="rgba(148, 163, 184, 0.28)",
            title_font=dict(color=THEME["muted"]),
            tickfont=dict(color=THEME["muted"]),
        ),
        yaxis=dict(
            gridcolor="rgba(148, 163, 184, 0.22)",
            zerolinecolor="rgba(148, 163, 184, 0.28)",
            title_font=dict(color=THEME["muted"]),
            tickfont=dict(color=THEME["muted"]),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0)",
            font=dict(color=THEME["muted"]),
        ),
    )

    return fig


def render_card(title, value, subtitle="", icon="📌", accent="#2563eb"):
    html = (
        f'<div class="card">'
        f'<div class="card-header">'
        f'<div class="card-icon" style="background:{accent}18; color:{accent};">{icon}</div>'
        f'<div class="card-title">{title}</div>'
        f'</div>'
        f'<div class="card-value">{value}</div>'
        f'<div class="card-subtitle">{subtitle}</div>'
        f'</div>'
    )

    st.markdown(html, unsafe_allow_html=True)


def render_forecast_card(title, value, category, icon, color, is_selected=False):
    border = color if is_selected else THEME["yellow_border"]

    html = (
        f'<div class="forecast-card" style="border-color:{border};">'
        f'<div class="forecast-title">{title}</div>'
        f'<div class="forecast-value">{round(value, 2)}</div>'
        f'<div class="forecast-badge" style="background:{color}18; color:{color};">'
        f'{icon} {category}'
        f'</div>'
        f'</div>'
    )

    st.markdown(html, unsafe_allow_html=True)


def render_pollutant_cards(pollutant_df: pd.DataFrame, selected_city: str):
    if pollutant_df.empty:
        st.info("Pollutant data is not available.")
        return

    cards_html = ""

    for _, row in pollutant_df.iterrows():
        pollutant = row["Pollutant"]
        short_name = row["Short"]
        value = row["Value"]
        unit = row["Unit"]
        icon = row["Icon"]
        color = row["Color"]

        cards_html += (
            f'<div class="pollutant-card" style="border-left-color:{color};">'
            f'<div class="pollutant-left">'
            f'<div class="pollutant-icon">{icon}</div>'
            f'<div>'
            f'<div class="pollutant-name">{pollutant}</div>'
            f'<div class="pollutant-short">({short_name})</div>'
            f'</div>'
            f'</div>'
            f'<div class="pollutant-right">'
            f'<div class="pollutant-value">{value}</div>'
            f'<div class="pollutant-unit">{unit}</div>'
            f'</div>'
            f'<div class="pollutant-arrow">›</div>'
            f'</div>'
        )

    section_html = (
        f'<div class="pollutant-section">'
        f'<div class="pollutant-section-header">'
        f'<div>'
        f'<div class="pollutant-section-title">Major Air Pollutants</div>'
        f'<div class="pollutant-section-city">{selected_city}</div>'
        f'</div>'
        f'<div class="pollutant-section-chip">Live breakdown</div>'
        f'</div>'
        f'<div class="pollutant-grid">{cards_html}</div>'
        f'</div>'
    )

    st.markdown(section_html, unsafe_allow_html=True)


def render_simple_prediction_explanation(driver_cards, predicted_aqi, category):
    cards_html = ""

    for card in driver_cards:
        cards_html += (
            f'<div class="driver-card">'
            f'<div class="driver-top">'
            f'<div class="driver-icon" style="background:{card["color"]}18; color:{card["color"]};">{card["icon"]}</div>'
            f'<div class="driver-title">{card["title"]}</div>'
            f'</div>'
            f'<div class="driver-text">{card["text"]}</div>'
            f'<div class="driver-badge" style="background:{card["color"]}15; color:{card["color"]};">{card["badge"]}</div>'
            f'</div>'
        )

    html = (
        f'<div class="driver-section">'
        f'<div class="driver-summary-card">'
        f'<div class="driver-summary-icon">🧠</div>'
        f'<div>'
        f'<div class="driver-summary-title">Why did the model predict this?</div>'
        f'<div class="driver-summary-text">'
        f'The forecast is mainly based on recent AQI, pollutant levels, pollution trend, and weather conditions. '
        f'For the selected city, the model predicts <b>{round(predicted_aqi, 1)}</b>, which falls under <b>{category}</b>.'
        f'</div>'
        f'</div>'
        f'</div>'
        f'<div class="driver-grid">{cards_html}</div>'
        f'</div>'
    )

    st.markdown(html, unsafe_allow_html=True)


def render_aqi_scale(current_aqi, predicted_aqi):
    current_position = (max(0, min(current_aqi, 500)) / 500) * 100
    forecast_position = (max(0, min(predicted_aqi, 500)) / 500) * 100

    html = (
        f'<div class="aqi-scale-card">'
        f'<div class="scale-track">'
        f'<div class="scale-marker current-marker" style="left:{current_position}%;">'
        f'<div class="marker-chip current-chip">Live {round(current_aqi, 1)}</div>'
        f'</div>'
        f'<div class="scale-marker forecast-marker" style="left:{forecast_position}%;">'
        f'<div class="marker-chip forecast-chip">Forecast {round(predicted_aqi, 1)}</div>'
        f'</div>'
        f'</div>'
        f'<div class="scale-labels">'
        f'<span>0</span><span>50</span><span>100</span><span>150</span><span>200</span><span>300</span><span>500</span>'
        f'</div>'
        f'<div class="scale-names">'
        f'<span>Good</span><span>Moderate</span><span>USG</span><span>Unhealthy</span><span>Very Unhealthy</span><span>Hazardous</span>'
        f'</div>'
        f'</div>'
    )

    st.markdown(html, unsafe_allow_html=True)


def render_hero(
    selected_city,
    selected_day_label,
    current_aqi,
    current_icon,
    current_category,
    predicted_aqi,
    lower_bound,
    upper_bound,
    trend_icon,
    trend_status,
    trend_color,
    category,
    icon,
    color,
    pm25_value,
    pm10_value,
    live_weather,
):
    weather_temp = live_weather.get("temperature_2m", "-") if live_weather else "-"
    weather_humidity = live_weather.get("relative_humidity_2m", "-") if live_weather else "-"
    weather_wind = live_weather.get("wind_speed_10m", "-") if live_weather else "-"
    weather_code = live_weather.get("weather_code", None) if live_weather else None
    weather_time = live_weather.get("time", "N/A") if live_weather else "N/A"
    weather_icon = get_weather_icon(weather_code)

    if weather_temp != "-":
        weather_temp = round(float(weather_temp), 1)

    if weather_humidity != "-":
        weather_humidity = round(float(weather_humidity), 1)

    if weather_wind != "-":
        weather_wind = round(float(weather_wind), 1)

    hero_html = (
        f'<div class="hero">'
        f'<div class="live-badge">● LIVE</div>'
        f'<div class="hero-title">{selected_city} Air Quality Index Forecast</div>'
        f'<div class="hero-subtitle">'
        f'Real-time AQI monitoring with machine learning based {selected_day_label.lower()} prediction.'
        f'</div>'
        f'</div>'
    )

    st.markdown(hero_html, unsafe_allow_html=True)

    hero_col1, hero_col2, hero_col3 = st.columns(3)

    with hero_col1:
        html = (
            f'<div class="hero-card">'
            f'<div class="hero-card-label">Live AQI</div>'
            f'<div class="hero-aqi-number">{round(current_aqi, 0)}</div>'
            f'<div class="hero-card-subtitle">AQI-US Standard · {current_icon} {current_category}</div>'
            f'<div class="hero-pollutants">'
            f'<span>PM2.5: {pm25_value}</span>'
            f'<span>PM10: {pm10_value}</span>'
            f'</div>'
            f'</div>'
        )
        st.markdown(html, unsafe_allow_html=True)

    with hero_col2:
        html = (
            f'<div class="hero-card">'
            f'<div class="hero-card-label">Forecast AQI is</div>'
            f'<div class="hero-risk" style="color:{color};">{icon} {category}</div>'
            f'<div class="hero-risk-text">'
            f'Predicted AQI: <b>{round(predicted_aqi, 2)}</b><br>'
            f'Expected range: <b>{round(lower_bound, 1)} – {round(upper_bound, 1)}</b><br>'
            f'Direction: <b style="color:{trend_color};">{trend_icon} {trend_status}</b>'
            f'</div>'
            f'</div>'
        )
        st.markdown(html, unsafe_allow_html=True)

    with hero_col3:
        html = (
            f'<div class="hero-card">'
            f'<div class="weather-icon">{weather_icon}</div>'
            f'<div class="weather-temp">{weather_temp}°C</div>'
            f'<div class="weather-subtitle">Live Weather Now</div>'
            f'<div class="weather-grid">'
            f'<div class="weather-item">'
            f'<div class="weather-label">Humidity</div>'
            f'<div class="weather-value">{weather_humidity}%</div>'
            f'</div>'
            f'<div class="weather-item">'
            f'<div class="weather-label">Wind</div>'
            f'<div class="weather-value">{weather_wind}</div>'
            f'</div>'
            f'<div class="weather-item">'
            f'<div class="weather-label">Source</div>'
            f'<div class="weather-value">Live</div>'
            f'</div>'
            f'</div>'
            f'<div class="weather-time">Updated: {str(weather_time)[:16]}</div>'
            f'</div>'
        )

        st.markdown(html, unsafe_allow_html=True)


def render_forecast_weather_section(future_weather_summary_df, future_weather_applied):
    if future_weather_summary_df.empty:
        st.info("Forecast weather data is not available right now. Saved weather features will be used instead.")
        return

    cards_html = ""

    for _, row in future_weather_summary_df.iterrows():
        cards_html += (
            f'<div class="forecast-weather-card">'
            f'<div class="forecast-weather-day">{row["Day"]}</div>'
            f'<div class="forecast-weather-temp">🌡️ {row["Avg Temp"]}°C</div>'
            f'<div class="forecast-weather-grid">'
            f'<div><span>Humidity</span><b>{row["Avg Humidity"]}%</b></div>'
            f'<div><span>Avg Wind</span><b>{row["Avg Wind"]}</b></div>'
            f'<div><span>Max Wind</span><b>{row["Max Wind"]}</b></div>'
            f'<div><span>Rain</span><b>{row["Rain Total"]}</b></div>'
            f'</div>'
            f'</div>'
        )

    source_text = "Live Open-Meteo Forecast API" if future_weather_applied else "Saved feature store values"

    html = (
        f'<div class="forecast-weather-section">'
        f'<div class="forecast-weather-header">'
        f'<div>'
        f'<div class="forecast-weather-title">Weather Used by ML Forecast</div>'
        f'<div class="forecast-weather-subtitle">These values are forecast weather inputs used by the AQI prediction model.</div>'
        f'</div>'
        f'<div class="forecast-weather-chip">{source_text}</div>'
        f'</div>'
        f'<div class="forecast-weather-cards">{cards_html}</div>'
        f'<div class="weather-note">Temperature can differ from Google because this dashboard uses Open-Meteo data and fixed UAE city coordinates.</div>'
        f'</div>'
    )

    st.markdown(html, unsafe_allow_html=True)


def build_city_comparison(df, model_bundle):
    rows = []

    for city in sorted(df["city"].unique().tolist()):
        city_df = df[df["city"] == city].dropna(subset=FEATURE_COLUMNS).copy()

        if city_df.empty:
            continue

        latest_row = city_df.iloc[-1]
        lat = float(latest_row["latitude"])
        lon = float(latest_row["longitude"])

        live_current = fetch_live_current_aqi(lat, lon)

        if live_current and live_current.get("us_aqi") is not None:
            current_aqi = float(live_current["us_aqi"])
        else:
            current_aqi = float(latest_row["us_aqi"])

        day1, _ = predict_for_target(model_bundle, "day_1", latest_row)
        day2, _ = predict_for_target(model_bundle, "day_2", latest_row)
        day3, _ = predict_for_target(model_bundle, "day_3", latest_row)

        avg_forecast = (day1 + day2 + day3) / 3
        category, _, icon, _ = get_aqi_category(avg_forecast)
        trend, change, trend_icon, _ = get_trend_status(current_aqi, day1)

        rows.append({
            "City": city,
            "Live AQI": round(current_aqi, 2),
            "Day 1": round(day1, 2),
            "Day 2": round(day2, 2),
            "Day 3": round(day3, 2),
            "3-Day Avg": round(avg_forecast, 2),
            "Risk": f"{icon} {category}",
            "Trend": f"{trend_icon} {trend}",
            "Change": change,
        })

    return pd.DataFrame(rows)


def create_aqi_trend_chart(chart_df, city):
    fig = px.line(
        chart_df,
        x="time",
        y="us_aqi",
        title=f"AQI Trend for {city}",
        markers=False,
    )

    fig.update_traces(line=dict(width=3, color=THEME["blue"]))
    fig.add_hrect(y0=0, y1=50, fillcolor="#22c55e", opacity=0.10, line_width=0)
    fig.add_hrect(y0=51, y1=100, fillcolor="#eab308", opacity=0.10, line_width=0)
    fig.add_hrect(y0=101, y1=150, fillcolor="#f97316", opacity=0.10, line_width=0)
    fig.add_hrect(y0=151, y1=200, fillcolor="#ef4444", opacity=0.10, line_width=0)

    return style_chart(fig, height=430)


def create_city_comparison_chart(comparison_df):
    fig = px.bar(
        comparison_df,
        x="City",
        y=["Live AQI", "Day 1", "Day 2", "Day 3"],
        barmode="group",
        title="Live AQI vs Forecast by City",
    )

    fig.update_layout(colorway=["#ca8a04", "#2563eb", "#0ea5e9", "#ef4444"])

    return style_chart(fig, height=470)


def create_feature_importance_chart(importance_df):
    importance_df = importance_df.copy()
    importance_df["Driver"] = importance_df["Feature"].apply(get_driver_label)

    fig = px.bar(
        importance_df.sort_values("Importance", ascending=True),
        x="Importance",
        y="Driver",
        orientation="h",
        title="Advanced Model Drivers",
        text="Importance",
        hover_data=["Feature"],
    )

    fig.update_traces(
        textposition="outside",
        marker_color=THEME["cyan"],
        marker_line_width=0,
    )

    return style_chart(fig, height=540)


def create_location_map(latitude, longitude, city):
    map_df = pd.DataFrame([{
        "city": city,
        "lat": latitude,
        "lon": longitude,
    }])

    fig = px.scatter_mapbox(
        map_df,
        lat="lat",
        lon="lon",
        hover_name="city",
        zoom=8,
        height=420,
    )

    fig.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=THEME["card_bg"],
        plot_bgcolor=THEME["card_bg"],
    )

    return fig


def render_model_performance_section(metrics, selected_target_key, selected_day_label):
    if not metrics:
        st.warning("metrics.json not found. Please run training_pipeline.py first.")
        return

    selected_target_metrics = metrics.get("targets", {}).get(selected_target_key, {})
    best_model = selected_target_metrics.get("best_model", "N/A")
    target_column = selected_target_metrics.get("target_column", "N/A")

    all_model_results = selected_target_metrics.get("all_model_results", {})
    best_result = all_model_results.get(best_model, {})

    best_rmse = best_result.get("rmse")
    best_mae = best_result.get("mae")
    best_r2 = best_result.get("r2")

    accuracy_label, accuracy_text, accuracy_color, accuracy_icon = get_accuracy_label(best_r2)

    rmse_text = "N/A"
    if best_rmse is not None:
        rmse_text = f"± {round(float(best_rmse), 1)} AQI"

    mae_text = "N/A"
    if best_mae is not None:
        mae_text = f"{round(float(best_mae), 1)} AQI"

    r2_text = "N/A"
    if best_r2 is not None:
        r2_text = f"{round(float(best_r2) * 100, 1)}%"

    model_html = (
        '<div class="model-section">'
        '<div class="model-section-header">'
        '<div>'
        '<div class="model-section-title">Model Performance</div>'
        '<div class="model-section-subtitle">Simple summary of how reliable this forecast model is.</div>'
        '</div>'
        '<div class="model-section-chip">ML Quality Check</div>'
        '</div>'

        '<div class="model-grid">'

        '<div class="model-card">'
        '<div class="model-card-icon" style="background:#2563eb18;color:#2563eb;">🤖</div>'
        '<div class="model-card-label">Best Model</div>'
        f'<div class="model-card-value">{clean_model_name(best_model)}</div>'
        '<div class="model-card-desc">This model performed best for the selected forecast.</div>'
        '</div>'

        '<div class="model-card">'
        f'<div class="model-card-icon" style="background:{accuracy_color}18;color:{accuracy_color};">{accuracy_icon}</div>'
        '<div class="model-card-label">Accuracy Level</div>'
        f'<div class="model-card-value" style="color:{accuracy_color};">{accuracy_label}</div>'
        f'<div class="model-card-desc">{accuracy_text}</div>'
        '</div>'

        '<div class="model-card">'
        '<div class="model-card-icon" style="background:#f9731618;color:#f97316;">🎯</div>'
        '<div class="model-card-label">Expected Error Range</div>'
        f'<div class="model-card-value">{rmse_text}</div>'
        '<div class="model-card-desc">Average uncertainty range around the forecast.</div>'
        '</div>'

        '<div class="model-card">'
        '<div class="model-card-icon" style="background:#0ea5e918;color:#0ea5e9;">📅</div>'
        '<div class="model-card-label">Forecast Target</div>'
        f'<div class="model-card-value">{clean_target_name(selected_target_key)}</div>'
        f'<div class="model-card-desc">Currently selected: {selected_day_label}</div>'
        '</div>'

        '</div>'

        '<div class="model-explain-row">'
        '<div class="model-explain-box">'
        '<b>RMSE</b><br>'
        f'The model prediction is usually around <b>{rmse_text}</b> away from the actual AQI.'
        '</div>'
        '<div class="model-explain-box">'
        '<b>MAE</b><br>'
        f'On average, the model misses the real AQI by about <b>{mae_text}</b>.'
        '</div>'
        '<div class="model-explain-box">'
        '<b>R² Score</b><br>'
        f'The model explains about <b>{r2_text}</b> of the AQI pattern for this forecast.'
        '</div>'
        '</div>'
        '</div>'
    )

    st.markdown(model_html, unsafe_allow_html=True)

    results_df = prepare_results_table(metrics, selected_target_key)

    if not results_df.empty:
        with st.expander("Advanced: View technical model comparison"):
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                fig_rmse = px.bar(
                    results_df.sort_values("RMSE", ascending=True),
                    x="Model",
                    y="RMSE",
                    title="Model Error Comparison",
                    text="RMSE",
                )
                fig_rmse.update_traces(textposition="outside", marker_color=THEME["blue"])
                st.plotly_chart(style_chart(fig_rmse, height=430), use_container_width=True)

            with chart_col2:
                fig_r2 = px.bar(
                    results_df.sort_values("R²", ascending=False),
                    x="Model",
                    y="R²",
                    title="Model Accuracy Comparison",
                    text="R²",
                )
                fig_r2.update_traces(textposition="outside", marker_color=THEME["cyan"])
                st.plotly_chart(style_chart(fig_r2, height=430), use_container_width=True)

            st.dataframe(results_df, use_container_width=True)
            st.caption(f"Training target column: {target_column}")


def render_hazard_alert(predicted_aqi, category):
    if predicted_aqi >= 300:
        st.error("🚨 Hazardous AQI Alert: Emergency air quality level. Stay indoors and avoid outdoor exposure.")
    elif predicted_aqi >= 200:
        st.warning("⚠️ Very Unhealthy AQI Alert: Avoid outdoor activity. Sensitive groups should stay indoors.")
    elif predicted_aqi >= 150:
        st.warning("⚠️ Unhealthy AQI Alert: Reduce outdoor activity and monitor sensitive groups.")
    else:
        st.success(f"Current forecast category: {category}")


def main():
    st.set_page_config(
        page_title="AQI Predictor Dashboard",
        page_icon="🌫️",
        layout="wide",
    )

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        .stApp {{
            background:
                radial-gradient(circle at top right, rgba(250, 204, 21, 0.13), transparent 30%),
                linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
        }}

        .block-container {{
            padding-top: 2.8rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }}

        h1, h2, h3 {{
            letter-spacing: -0.035em !important;
            color: {THEME["text"]};
        }}

        .app-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 18px;
            margin-bottom: 26px;
            padding-top: 12px;
        }}

        .brand-wrap {{
            display: flex;
            align-items: center;
            gap: 13px;
        }}

        .brand-mark {{
            width: 48px;
            height: 48px;
            border-radius: 15px;
            background: linear-gradient(135deg, #60a5fa, #2563eb);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
            font-weight: 950;
            box-shadow: 0 12px 26px rgba(37, 99, 235, 0.25);
        }}

        .brand-title {{
            font-size: 31px;
            font-weight: 950;
            letter-spacing: -0.05em;
            color: {THEME["text"]};
        }}

        .header-tags {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            justify-content: flex-end;
        }}

        .header-tag {{
            padding: 10px 14px;
            border-radius: 999px;
            background: {THEME["card_bg"]};
            border: 1px solid {THEME["yellow_border"]};
            color: #334155;
            font-size: 13px;
            font-weight: 850;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }}

        .hero {{
            padding: 34px 42px;
            border-radius: 30px;
            background:
                radial-gradient(circle at 70% 45%, rgba(255, 211, 68, 0.5), transparent 20%),
                linear-gradient(135deg, #fff8d8 0%, #fff1a6 48%, #facc15 100%);
            border: 1px solid #fde68a;
            box-shadow: 0 24px 54px rgba(15, 23, 42, 0.10);
            margin-bottom: 22px;
        }}

        .live-badge {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 14px;
            border-radius: 999px;
            background: #ef4444;
            color: white;
            font-size: 12px;
            font-weight: 950;
            margin-bottom: 20px;
        }}

        .hero-title {{
            font-size: 38px;
            line-height: 1.05;
            font-weight: 950;
            color: {THEME["text"]};
            letter-spacing: -0.05em;
            margin-bottom: 12px;
        }}

        .hero-subtitle {{
            color: #475569;
            font-size: 17px;
            font-weight: 850;
        }}

        .hero-card {{
            min-height: 292px;
            padding: 30px;
            border-radius: 26px;
            background: rgba(255, 253, 242, 0.96);
            border: 1px solid rgba(243, 232, 162, 0.95);
            box-shadow: 0 20px 44px rgba(15, 23, 42, 0.10);
            color: {THEME["text"]};
        }}

        .hero-card-label {{
            color: #334155;
            font-size: 15px;
            font-weight: 950;
            margin-bottom: 14px;
        }}

        .hero-aqi-number {{
            font-size: 88px;
            font-weight: 950;
            line-height: 0.95;
            letter-spacing: -0.06em;
            color: #ca8a04;
        }}

        .hero-card-subtitle {{
            color: #64748b;
            font-size: 15px;
            font-weight: 900;
            line-height: 1.35;
            margin-top: 16px;
        }}

        .hero-pollutants {{
            display: flex;
            flex-wrap: wrap;
            gap: 22px;
            margin-top: 28px;
            color: #334155;
            font-size: 17px;
            font-weight: 950;
        }}

        .hero-risk {{
            font-size: 31px;
            line-height: 1.15;
            font-weight: 950;
            letter-spacing: -0.04em;
            margin-bottom: 20px;
        }}

        .hero-risk-text {{
            color: #475569;
            font-size: 16px;
            font-weight: 850;
            line-height: 1.7;
        }}

        .weather-icon {{
            font-size: 43px;
            margin-bottom: 12px;
        }}

        .weather-temp {{
            font-size: 50px;
            font-weight: 950;
            color: {THEME["text"]};
            line-height: 1;
            letter-spacing: -0.05em;
            margin-bottom: 12px;
        }}

        .weather-subtitle {{
            color: #475569;
            font-weight: 900;
            font-size: 16px;
            line-height: 1.35;
        }}

        .weather-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 13px;
            margin-top: 27px;
        }}

        .weather-item {{
            background: rgba(255, 255, 255, 0.72);
            border-radius: 18px;
            padding: 15px 10px;
            text-align: center;
            border: 1px solid #e5e7eb;
            min-height: 76px;
        }}

        .weather-label {{
            color: #64748b;
            font-size: 12px;
            font-weight: 950;
            margin-bottom: 7px;
        }}

        .weather-value {{
            color: {THEME["text"]};
            font-size: 16px;
            font-weight: 950;
            word-break: break-word;
        }}

        .weather-time {{
            margin-top: 16px;
            color: #64748b;
            font-size: 12px;
            font-weight: 750;
        }}

        .source-strip {{
            padding: 18px 20px;
            border-radius: 22px;
            background: {THEME["card_bg"]};
            border: 1px solid {THEME["yellow_border"]};
            margin-top: 24px;
            color: #475569;
            font-size: 14px;
            font-weight: 780;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.05);
        }}

        .section-title {{
            font-size: 27px;
            font-weight: 950;
            color: {THEME["text"]};
            margin-top: 40px;
            margin-bottom: 17px;
            letter-spacing: -0.04em;
        }}

        .card, .info-card, .aqi-scale-card, .pollutant-section, .driver-section, .model-section, .forecast-weather-section {{
            background: {THEME["card_bg"]};
            border: 1px solid {THEME["yellow_border"]};
            box-shadow: 0 16px 38px rgba(15, 23, 42, 0.07);
        }}

        .card {{
            padding: 22px;
            border-radius: 24px;
            min-height: 156px;
            color: {THEME["text"]};
        }}

        .card-header {{
            display: flex;
            align-items: center;
            gap: 11px;
            margin-bottom: 15px;
        }}

        .card-icon {{
            width: 40px;
            height: 40px;
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            font-weight: 950;
        }}

        .card-title {{
            color: {THEME["muted"]};
            font-size: 14px;
            font-weight: 950;
        }}

        .card-value {{
            color: {THEME["text"]};
            font-size: 34px;
            line-height: 1.05;
            font-weight: 950;
            margin-bottom: 10px;
            word-break: break-word;
        }}

        .card-subtitle {{
            color: {THEME["muted"]};
            font-size: 13px;
            font-weight: 750;
        }}

        .forecast-card {{
            padding: 22px;
            border-radius: 24px;
            border: 2px solid {THEME["yellow_border"]};
            min-height: 166px;
            background: {THEME["card_bg"]};
            box-shadow: 0 16px 38px rgba(15,23,42,0.07);
            color: {THEME["text"]};
        }}

        .forecast-title {{
            color: {THEME["muted"]};
            font-size: 14px;
            font-weight: 950;
            margin-bottom: 11px;
        }}

        .forecast-value {{
            color: {THEME["text"]};
            font-size: 38px;
            font-weight: 950;
            margin-bottom: 13px;
            line-height: 1;
        }}

        .forecast-badge {{
            display: inline-block;
            padding: 8px 11px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 950;
        }}

        .info-card {{
            padding: 24px;
            border-radius: 24px;
            min-height: 168px;
            color: {THEME["text"]};
        }}

        .info-card h4 {{
            margin-top: 0;
            color: {THEME["text"]};
            font-weight: 950;
            letter-spacing: -0.03em;
        }}

        .aqi-scale-card {{
            padding: 30px;
            border-radius: 26px;
        }}

        .scale-track {{
            position: relative;
            height: 14px;
            border-radius: 999px;
            background: linear-gradient(
                to right,
                #22c55e 0%, #22c55e 10%,
                #eab308 10%, #eab308 20%,
                #f97316 20%, #f97316 30%,
                #ef4444 30%, #ef4444 40%,
                #8b5cf6 40%, #8b5cf6 60%,
                #7f1d1d 60%, #7f1d1d 100%
            );
            margin-top: 42px;
        }}

        .scale-marker {{
            position: absolute;
            top: -22px;
            transform: translateX(-50%);
            width: 3px;
            height: 56px;
            border-radius: 999px;
        }}

        .current-marker {{
            background: #2563eb;
        }}

        .forecast-marker {{
            background: #111827;
        }}

        .marker-chip {{
            position: absolute;
            top: -34px;
            left: 50%;
            transform: translateX(-50%);
            color: white;
            padding: 6px 11px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 950;
            white-space: nowrap;
        }}

        .current-chip {{
            background: #2563eb;
        }}

        .forecast-chip {{
            background: #111827;
        }}

        .scale-labels, .scale-names {{
            display: flex;
            justify-content: space-between;
        }}

        .scale-labels {{
            margin-top: 21px;
            color: #64748b;
            font-size: 13px;
            font-weight: 850;
        }}

        .scale-names {{
            margin-top: 8px;
            color: #475569;
            font-size: 12px;
            font-weight: 950;
        }}

        .forecast-weather-section {{
            padding: 28px;
            border-radius: 26px;
        }}

        .forecast-weather-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 16px;
            margin-bottom: 22px;
        }}

        .forecast-weather-title {{
            font-size: 25px;
            font-weight: 950;
            color: {THEME["text"]};
            letter-spacing: -0.04em;
        }}

        .forecast-weather-subtitle {{
            margin-top: 6px;
            color: #64748b;
            font-size: 15px;
            font-weight: 750;
        }}

        .forecast-weather-chip {{
            padding: 10px 14px;
            border-radius: 999px;
            border: 1px solid #bfdbfe;
            background: #eff6ff;
            color: #2563eb;
            font-weight: 950;
            font-size: 13px;
            white-space: nowrap;
        }}

        .forecast-weather-cards {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 18px;
        }}

        .forecast-weather-card {{
            padding: 22px;
            border-radius: 22px;
            background: #f8fafc;
            border: 1px solid #eef2f7;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }}

        .forecast-weather-day {{
            color: #64748b;
            font-size: 14px;
            font-weight: 950;
            margin-bottom: 10px;
        }}

        .forecast-weather-temp {{
            color: {THEME["text"]};
            font-size: 30px;
            font-weight: 950;
            margin-bottom: 18px;
        }}

        .forecast-weather-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
        }}

        .forecast-weather-grid div {{
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 12px;
        }}

        .forecast-weather-grid span {{
            display: block;
            color: #64748b;
            font-size: 12px;
            font-weight: 850;
            margin-bottom: 5px;
        }}

        .forecast-weather-grid b {{
            color: {THEME["text"]};
            font-size: 16px;
            font-weight: 950;
        }}

        .weather-note {{
            margin-top: 18px;
            padding: 14px 16px;
            border-radius: 16px;
            background: #eff6ff;
            color: #1e40af;
            font-size: 14px;
            font-weight: 750;
            border: 1px solid #bfdbfe;
        }}

        .pollutant-section {{
            padding: 28px;
            border-radius: 26px;
        }}

        .pollutant-section-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 22px;
            gap: 16px;
        }}

        .pollutant-section-title {{
            font-size: 25px;
            font-weight: 950;
            color: {THEME["text"]};
            letter-spacing: -0.04em;
        }}

        .pollutant-section-city {{
            margin-top: 4px;
            font-size: 17px;
            color: #2563eb;
            font-weight: 800;
        }}

        .pollutant-section-chip {{
            padding: 10px 14px;
            border-radius: 999px;
            border: 1px solid #bfdbfe;
            background: #eff6ff;
            color: #2563eb;
            font-weight: 900;
            font-size: 13px;
            white-space: nowrap;
        }}

        .pollutant-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 18px;
        }}

        .pollutant-card {{
            min-height: 110px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            padding: 20px 18px;
            border-radius: 18px;
            background: #f8fafc;
            border: 1px solid #eef2f7;
            border-left: 5px solid #22c55e;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }}

        .pollutant-left {{
            display: flex;
            align-items: center;
            gap: 15px;
            min-width: 0;
        }}

        .pollutant-icon {{
            width: 46px;
            height: 46px;
            border-radius: 15px;
            background: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            border: 1px solid #e5e7eb;
            flex-shrink: 0;
        }}

        .pollutant-name {{
            color: {THEME["text"]};
            font-size: 15px;
            font-weight: 950;
            line-height: 1.2;
        }}

        .pollutant-short {{
            margin-top: 3px;
            color: #334155;
            font-size: 14px;
            font-weight: 800;
        }}

        .pollutant-right {{
            text-align: right;
            flex-shrink: 0;
        }}

        .pollutant-value {{
            color: {THEME["text"]};
            font-size: 25px;
            font-weight: 950;
            line-height: 1;
        }}

        .pollutant-unit {{
            margin-top: 5px;
            color: #334155;
            font-size: 13px;
            font-weight: 850;
        }}

        .pollutant-arrow {{
            color: #94a3b8;
            font-size: 36px;
            line-height: 1;
            flex-shrink: 0;
        }}

        .driver-section {{
            padding: 28px;
            border-radius: 26px;
        }}

        .driver-summary-card {{
            display: flex;
            gap: 18px;
            padding: 24px;
            border-radius: 24px;
            background: linear-gradient(135deg, #eff6ff 0%, #fffdf2 100%);
            border: 1px solid #bfdbfe;
            margin-bottom: 22px;
            align-items: flex-start;
        }}

        .driver-summary-icon {{
            width: 54px;
            height: 54px;
            border-radius: 18px;
            background: #2563eb;
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
            flex-shrink: 0;
            box-shadow: 0 12px 26px rgba(37, 99, 235, 0.25);
        }}

        .driver-summary-title {{
            font-size: 24px;
            font-weight: 950;
            color: {THEME["text"]};
            letter-spacing: -0.04em;
            margin-bottom: 8px;
        }}

        .driver-summary-text {{
            color: #475569;
            font-size: 15px;
            font-weight: 700;
            line-height: 1.7;
        }}

        .driver-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 18px;
        }}

        .driver-card {{
            padding: 22px;
            border-radius: 22px;
            background: #f8fafc;
            border: 1px solid #eef2f7;
            min-height: 205px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }}

        .driver-top {{
            display: flex;
            align-items: center;
            gap: 13px;
            margin-bottom: 14px;
        }}

        .driver-icon {{
            width: 46px;
            height: 46px;
            border-radius: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            flex-shrink: 0;
        }}

        .driver-title {{
            color: {THEME["text"]};
            font-size: 17px;
            font-weight: 950;
            line-height: 1.25;
        }}

        .driver-text {{
            color: #475569;
            font-size: 14px;
            font-weight: 700;
            line-height: 1.65;
            margin-bottom: 16px;
        }}

        .driver-badge {{
            display: inline-flex;
            padding: 8px 12px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 950;
        }}

        .model-section {{
            padding: 28px;
            border-radius: 26px;
        }}

        .model-section-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 16px;
            margin-bottom: 22px;
        }}

        .model-section-title {{
            font-size: 27px;
            font-weight: 950;
            color: {THEME["text"]};
            letter-spacing: -0.04em;
        }}

        .model-section-subtitle {{
            margin-top: 6px;
            color: #64748b;
            font-size: 15px;
            font-weight: 750;
        }}

        .model-section-chip {{
            padding: 10px 14px;
            border-radius: 999px;
            border: 1px solid #bfdbfe;
            background: #eff6ff;
            color: #2563eb;
            font-weight: 950;
            font-size: 13px;
            white-space: nowrap;
        }}

        .model-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 18px;
            margin-bottom: 20px;
        }}

        .model-card {{
            padding: 22px;
            border-radius: 22px;
            background: #f8fafc;
            border: 1px solid #eef2f7;
            min-height: 190px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }}

        .model-card-icon {{
            width: 46px;
            height: 46px;
            border-radius: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 23px;
            margin-bottom: 16px;
        }}

        .model-card-label {{
            color: #64748b;
            font-size: 13px;
            font-weight: 950;
            margin-bottom: 8px;
        }}

        .model-card-value {{
            color: {THEME["text"]};
            font-size: 24px;
            line-height: 1.15;
            font-weight: 950;
            letter-spacing: -0.035em;
            margin-bottom: 12px;
        }}

        .model-card-desc {{
            color: #64748b;
            font-size: 13px;
            font-weight: 750;
            line-height: 1.5;
        }}

        .model-explain-row {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 14px;
        }}

        .model-explain-box {{
            padding: 16px 18px;
            border-radius: 18px;
            background: #fff8d8;
            border: 1px solid #fde68a;
            color: #475569;
            font-size: 14px;
            line-height: 1.6;
        }}

        .model-explain-box b {{
            color: {THEME["text"]};
        }}

        .footer {{
            margin-top: 45px;
            padding: 22px;
            border-radius: 22px;
            background: #0f172a;
            color: rgba(255,255,255,0.82);
            font-size: 14px;
            text-align: center;
        }}

        section[data-testid="stSidebar"] {{
            background: #ffffff;
            border-right: 1px solid #e5e7eb;
        }}

        section[data-testid="stSidebar"] label {{
            font-size: 15px !important;
            padding-left: 8px !important;
            font-weight: 750 !important;
        }}

        div[data-testid="stDataFrame"] {{
            border-radius: 16px;
            overflow: hidden;
        }}

        @media (max-width: 1200px) {{
            .model-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}

        @media (max-width: 1100px) {{
            .pollutant-grid, .driver-grid, .forecast-weather-cards {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}

        @media (max-width: 800px) {{
            .app-header {{
                flex-direction: column;
                align-items: flex-start;
            }}

            .header-tags {{
                justify-content: flex-start;
            }}

            .hero {{
                padding: 28px;
            }}

            .hero-title {{
                font-size: 30px;
            }}

            .pollutant-grid, .driver-grid, .model-grid, .model-explain-row, .forecast-weather-cards {{
                grid-template-columns: 1fr;
            }}

            .pollutant-section-header, .driver-summary-card, .model-section-header, .forecast-weather-header {{
                flex-direction: column;
                align-items: flex-start;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    df, feature_store_source = load_feature_store()
    df = create_city_column(df)

    model_bundle = load_model_bundle()
    metrics = load_metrics()

    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        st.error(f"Missing feature columns in feature store: {missing_features}")
        st.stop()

    cities = sorted(df["city"].dropna().unique().tolist())
    cities = [city for city in cities if str(city).lower() != "none"]

    st.sidebar.title("Dashboard Controls")

    selected_city = st.sidebar.radio(
        "Select City",
        cities,
        index=cities.index("Dubai") if "Dubai" in cities else 0,
    )

    selected_day_label = st.sidebar.radio(
        "Select Forecast Day",
        list(DAY_OPTIONS.keys()),
    )

    selected_target_key = DAY_OPTIONS[selected_day_label]

    location_df = df[df["city"] == selected_city].copy()
    clean_df = location_df.dropna(subset=FEATURE_COLUMNS).copy()

    if clean_df.empty:
        st.error("No valid feature rows found for this city.")
        st.stop()

    latest_row = clean_df.iloc[-1]
    latest_latitude = float(latest_row["latitude"])
    latest_longitude = float(latest_row["longitude"])

    live_weather = fetch_live_current_weather(latest_latitude, latest_longitude)

    future_weather_df = fetch_future_weather_forecast(
        latitude=latest_latitude,
        longitude=latest_longitude,
    )

    latest_row, future_weather_applied = update_latest_row_with_future_weather(
        latest_row=latest_row,
        future_weather_df=future_weather_df,
    )

    future_weather_summary_df = get_future_weather_summary(future_weather_df)

    live_current = fetch_live_current_aqi(latest_latitude, latest_longitude)

    if live_current and live_current.get("us_aqi") is not None:
        current_aqi = float(live_current["us_aqi"])
        current_source = "Live Open-Meteo current AQI"
        live_time = live_current.get("time", None)
    else:
        current_aqi = float(latest_row["us_aqi"])
        current_source = "Saved feature store AQI"
        live_time = None

    latest_time = latest_row.get("time", "N/A")

    day_predictions = {}

    for label, key in DAY_OPTIONS.items():
        prediction, model_name = predict_for_target(model_bundle, key, latest_row)
        category, message, icon, color = get_aqi_category(prediction)

        day_predictions[key] = {
            "label": label,
            "prediction": prediction,
            "model_name": model_name,
            "category": category,
            "message": message,
            "icon": icon,
            "color": color,
        }

    selected_prediction = day_predictions[selected_target_key]

    predicted_aqi = selected_prediction["prediction"]
    category = selected_prediction["category"]
    message = selected_prediction["message"]
    icon = selected_prediction["icon"]
    color = selected_prediction["color"]
    selected_model_name = selected_prediction["model_name"]

    current_category, _, current_icon, current_color = get_aqi_category(current_aqi)
    trend_status, trend_change, trend_icon, trend_color = get_trend_status(current_aqi, predicted_aqi)

    model_metrics = get_selected_model_metrics(metrics, selected_target_key, selected_model_name)
    selected_rmse = float(model_metrics["rmse"]) if model_metrics and "rmse" in model_metrics else None

    if selected_rmse:
        lower_bound = max(0, predicted_aqi - selected_rmse)
        upper_bound = min(500, predicted_aqi + selected_rmse)
    else:
        lower_bound = max(0, predicted_aqi - 15)
        upper_bound = min(500, predicted_aqi + 15)

    pollutant_df = make_pollutant_table(latest_row, live_current=live_current)

    pm25_value = "-"
    pm10_value = "-"

    if not pollutant_df.empty:
        pm25_match = pollutant_df[pollutant_df["Short"] == "PM2.5"]
        pm10_match = pollutant_df[pollutant_df["Short"] == "PM10"]

        if not pm25_match.empty:
            pm25_value = pm25_match["Value"].iloc[0]

        if not pm10_match.empty:
            pm10_value = pm10_match["Value"].iloc[0]

    st.markdown(
        '<div class="app-header">'
        '<div class="brand-wrap">'
        '<div class="brand-mark">AQ</div>'
        '<div class="brand-title">AQI Forecast</div>'
        '</div>'
        '<div class="header-tags">'
        '<div class="header-tag">AQI-US Standard</div>'
        '<div class="header-tag">UAE Cities</div>'
        '<div class="header-tag">Hopsworks Feature Store</div>'
        '<div class="header-tag">ML Forecast</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    render_hero(
        selected_city=selected_city,
        selected_day_label=selected_day_label,
        current_aqi=current_aqi,
        current_icon=current_icon,
        current_category=current_category,
        predicted_aqi=predicted_aqi,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        trend_icon=trend_icon,
        trend_status=trend_status,
        trend_color=trend_color,
        category=category,
        icon=icon,
        color=color,
        pm25_value=pm25_value,
        pm10_value=pm10_value,
        live_weather=live_weather,
    )

    source_html = (
        f'<div class="source-strip">'
        f'<b>Feature Store Source:</b> {feature_store_source}'
        f'<br>'
        f'<b>Model Source:</b> Local model backup · models/aqi_model.joblib'
        f'<br>'
        f'<b>Live AQI source:</b> {current_source}'
        f'{" · <b>Live AQI timestamp:</b> " + str(live_time) if live_time else ""}'
        f'<br>'
        f'<b>Live weather source:</b> Open-Meteo current weather'
        f'<br>'
        f'<b>Forecast weather source:</b> {"Live Open-Meteo Forecast API" if future_weather_applied else "Saved feature store values"}'
        f'<br>'
        f'<b>Model feature timestamp:</b> {str(latest_time)[:16]}'
        f'</div>'
    )

    st.markdown(source_html, unsafe_allow_html=True)

    if is_feature_store_old(latest_time, days=3):
        st.warning("⚠️ Feature store data is older than 3 days. Run feature_pipeline.py for fresher historical AQI features.")

    render_hazard_alert(predicted_aqi, category)
    st.info(message)

    st.markdown('<div class="section-title">Forecast Summary</div>', unsafe_allow_html=True)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        render_card("Live Current AQI", round(current_aqi, 2), current_category, current_icon, current_color)

    with kpi2:
        render_card(selected_day_label, round(predicted_aqi, 2), f"{trend_icon} {trend_status} ({trend_change})", "📈", trend_color)

    with kpi3:
        render_card("Forecast Category", f"{icon} {category}", "Predicted AQI risk level", "🧭", color)

    with kpi4:
        render_card("Prediction Range", f"{round(lower_bound, 1)} – {round(upper_bound, 1)}", "Estimated using model RMSE", "🎯", "#2563eb")

    st.markdown('<div class="section-title">AQI Scale</div>', unsafe_allow_html=True)
    render_aqi_scale(current_aqi, predicted_aqi)

    st.markdown('<div class="section-title">3-Day Forecast Overview</div>', unsafe_allow_html=True)

    card_col1, card_col2, card_col3, card_col4 = st.columns(4)

    with card_col1:
        item = day_predictions["day_1"]
        render_forecast_card("Day 1 Forecast", item["prediction"], item["category"], item["icon"], item["color"], selected_target_key == "day_1")

    with card_col2:
        item = day_predictions["day_2"]
        render_forecast_card("Day 2 Forecast", item["prediction"], item["category"], item["icon"], item["color"], selected_target_key == "day_2")

    with card_col3:
        item = day_predictions["day_3"]
        render_forecast_card("Day 3 Forecast", item["prediction"], item["category"], item["icon"], item["color"], selected_target_key == "day_3")

    with card_col4:
        item = day_predictions["next_72h_avg"]
        render_forecast_card("3-Day Average", item["prediction"], item["category"], item["icon"], item["color"], selected_target_key == "next_72h_avg")

    st.markdown('<div class="section-title">Weather Inputs</div>', unsafe_allow_html=True)
    render_forecast_weather_section(future_weather_summary_df, future_weather_applied)

    st.markdown('<div class="section-title">Health Recommendations</div>', unsafe_allow_html=True)

    recommendations = get_health_recommendations(predicted_aqi)
    rec_cols = st.columns(3)

    for index, recommendation in enumerate(recommendations):
        rec_html = (
            f'<div class="info-card">'
            f'<h4>Recommendation {index + 1}</h4>'
            f'<p>{recommendation}</p>'
            f'</div>'
        )

        with rec_cols[index % 3]:
            st.markdown(rec_html, unsafe_allow_html=True)

    st.markdown('<div class="section-title">AQI Trend</div>', unsafe_allow_html=True)

    chart_df = clean_df[["time", "us_aqi"]].tail(240).copy()
    chart_df = chart_df.drop_duplicates(subset=["time"], keep="last")
    st.plotly_chart(create_aqi_trend_chart(chart_df, selected_city), use_container_width=True)

    st.markdown('<div class="section-title">Major Air Pollutants</div>', unsafe_allow_html=True)
    render_pollutant_cards(pollutant_df, selected_city)

    st.markdown('<div class="section-title">City Comparison</div>', unsafe_allow_html=True)

    comparison_df = build_city_comparison(df, model_bundle)

    if not comparison_df.empty:
        st.plotly_chart(create_city_comparison_chart(comparison_df), use_container_width=True)

        best_city_row = comparison_df.sort_values("Live AQI", ascending=True).iloc[0]
        worst_city_row = comparison_df.sort_values("Live AQI", ascending=False).iloc[0]

        insight_col1, insight_col2 = st.columns(2)

        with insight_col1:
            st.success(f"Best current air quality: {best_city_row['City']} with AQI {best_city_row['Live AQI']}")

        with insight_col2:
            st.error(f"Highest current AQI: {worst_city_row['City']} with AQI {worst_city_row['Live AQI']}")

        with st.expander("View city comparison table"):
            st.dataframe(comparison_df, use_container_width=True)

    st.markdown('<div class="section-title">Selected City Map</div>', unsafe_allow_html=True)

    st.plotly_chart(
        create_location_map(latest_latitude, latest_longitude, selected_city),
        use_container_width=True,
    )

    st.markdown('<div class="section-title">Why This Prediction?</div>', unsafe_allow_html=True)

    importance_df = get_feature_importance(model_bundle, selected_target_key, clean_df, metrics)

    if not importance_df.empty:
        driver_cards = build_simple_driver_cards(
            importance_df=importance_df,
            current_aqi=current_aqi,
            predicted_aqi=predicted_aqi,
            pollutant_df=pollutant_df,
            future_weather_applied=future_weather_applied,
        )

        render_simple_prediction_explanation(driver_cards, predicted_aqi, category)

        with st.expander("Advanced: View technical model drivers"):
            st.plotly_chart(create_feature_importance_chart(importance_df), use_container_width=True)

            advanced_df = importance_df.copy()
            advanced_df["Simple Meaning"] = advanced_df["Feature"].apply(get_driver_label)
            st.dataframe(advanced_df, use_container_width=True)
    else:
        driver_cards = build_simple_driver_cards(
            importance_df=pd.DataFrame(),
            current_aqi=current_aqi,
            predicted_aqi=predicted_aqi,
            pollutant_df=pollutant_df,
            future_weather_applied=future_weather_applied,
        )

        render_simple_prediction_explanation(driver_cards, predicted_aqi, category)

    render_model_performance_section(metrics, selected_target_key, selected_day_label)

    st.markdown('<div class="section-title">Downloads</div>', unsafe_allow_html=True)

    latest_prediction_df = pd.DataFrame([{
        "city": selected_city,
        "forecast": selected_day_label,
        "feature_store_source": feature_store_source,
        "model_source": "Local model backup",
        "live_current_aqi": current_aqi,
        "live_current_source": current_source,
        "live_weather_source": "Open-Meteo current weather",
        "future_weather_applied": future_weather_applied,
        "feature_timestamp": str(latest_time),
        "predicted_aqi": predicted_aqi,
        "prediction_lower_bound": lower_bound,
        "prediction_upper_bound": upper_bound,
        "direction": trend_status,
        "category": category,
        "model": selected_model_name,
    }])

    download_col1, download_col2 = st.columns(2)

    with download_col1:
        st.download_button(
            label="Download Latest Prediction CSV",
            data=latest_prediction_df.to_csv(index=False),
            file_name=f"{selected_city.lower().replace(' ', '_')}_{selected_target_key}_aqi_prediction.csv",
            mime="text/csv",
        )

    with download_col2:
        if metrics:
            st.download_button(
                label="Download Model Metrics JSON",
                data=json.dumps(metrics, indent=4),
                file_name="model_metrics.json",
                mime="application/json",
            )

    with st.expander("Advanced: Latest Model Feature Values"):
        latest_feature_df = latest_row[FEATURE_COLUMNS].to_frame(name="Latest Value")
        st.dataframe(latest_feature_df, use_container_width=True)

    with st.expander("Advanced: Recent Saved Raw Data"):
        st.dataframe(clean_df.tail(50), use_container_width=True)

    with st.expander("Technical Model Summary"):
        st.write(f"**Feature Store Source:** {feature_store_source}")
        st.write("**Model Source:** Local model backup")
        st.write(f"**AQI Standard:** AQI-US")
        st.write(f"**Number of Features:** {len(FEATURE_COLUMNS)}")
        st.write(f"**Selected Forecast:** {selected_day_label}")
        st.write(f"**Selected Model:** {selected_model_name}")
        st.write(f"**Feature Store Rows for Selected City:** {len(clean_df)}")
        st.write(f"**Live AQI Source:** {current_source}")
        st.write("**Live Weather Source:** Open-Meteo current weather")
        st.write(f"**Future Weather Applied:** {future_weather_applied}")
        st.write("**Inference Features:** Hopsworks/local historical AQI + pollutant history + live future weather forecast")
        st.write(f"**Model Feature Timestamp:** {str(latest_time)[:16]}")

    st.markdown(
        '<div class="footer">'
        'Built with Python, Streamlit, Scikit-learn, Plotly, Open-Meteo APIs, and Hopsworks Feature Store. '
        'AQI standard: AQI-US. Model source: local backup with Model Registry save attempt.'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()