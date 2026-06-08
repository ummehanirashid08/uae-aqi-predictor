import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config import FEATURE_STORE_PATH
from hopsworks_store import save_features_to_hopsworks


CITY_LOCATIONS = {
    "Abu Dhabi": {"latitude": 24.4539, "longitude": 54.3773},
    "Ajman": {"latitude": 25.4052, "longitude": 55.5136},
    "Al Ain": {"latitude": 24.1302, "longitude": 55.8023},
    "Dubai": {"latitude": 25.2048, "longitude": 55.2708},
    "Ras Al Khaimah": {"latitude": 25.8007, "longitude": 55.9762},
    "Sharjah": {"latitude": 25.3463, "longitude": 55.4209},
}


AIR_QUALITY_HOURLY_COLUMNS = [
    "us_aqi",
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "dust",
]


WEATHER_HOURLY_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
]


POLLUTANT_COLUMNS = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "dust",
]


LAG_COLUMNS = [
    "us_aqi",
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "dust",
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
]


ROLLING_COLUMNS = [
    "us_aqi",
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "dust",
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
]


LAG_HOURS = [1, 3, 6, 12, 24]
ROLLING_WINDOWS = [3, 6, 12, 24, 48]


def parse_args():
    parser = argparse.ArgumentParser(description="UAE AQI feature pipeline")

    parser.add_argument("--latitude", type=float, default=None)
    parser.add_argument("--longitude", type=float, default=None)
    parser.add_argument("--timezone", type=str, default="Asia/Dubai")
    parser.add_argument("--past-days", type=int, default=10)
    parser.add_argument("--forecast-days", type=int, default=3)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--skip-hopsworks", action="store_true")

    return parser.parse_args()


def safe_request_json(url: str, params: dict, timeout: int = 45):
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()

    except Exception as error:
        print(f"Request failed for {url}")
        print(error)
        return None


def fetch_air_quality_data(
    latitude: float,
    longitude: float,
    timezone: str = "Asia/Dubai",
    past_days: int = 10,
    forecast_days: int = 3,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone,
        "hourly": ",".join(AIR_QUALITY_HOURLY_COLUMNS),
    }

    if start_date and end_date:
        params["start_date"] = start_date
        params["end_date"] = end_date
    else:
        params["past_days"] = past_days
        params["forecast_days"] = forecast_days

    data = safe_request_json(url, params)

    if not data or "hourly" not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data["hourly"])

    if df.empty:
        return df

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["latitude"] = round(float(latitude), 4)
    df["longitude"] = round(float(longitude), 4)

    return df


def fetch_weather_data(
    latitude: float,
    longitude: float,
    timezone: str = "Asia/Dubai",
    past_days: int = 10,
    forecast_days: int = 3,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    url = "https://api.open-meteo.com/v1/forecast"

    hourly_columns = WEATHER_HOURLY_COLUMNS.copy()

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone,
        "hourly": ",".join(hourly_columns),
    }

    if start_date and end_date:
        params["start_date"] = start_date
        params["end_date"] = end_date
    else:
        params["past_days"] = past_days
        params["forecast_days"] = forecast_days

    data = safe_request_json(url, params)

    if not data or "hourly" not in data:
        return pd.DataFrame()

    df = pd.DataFrame(data["hourly"])

    if df.empty:
        return df

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["latitude"] = round(float(latitude), 4)
    df["longitude"] = round(float(longitude), 4)

    return df


def fetch_city_data(
    city: str,
    latitude: float,
    longitude: float,
    timezone: str,
    past_days: int,
    forecast_days: int,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    print(f"Fetching data for {city}...")

    aqi_df = fetch_air_quality_data(
        latitude=latitude,
        longitude=longitude,
        timezone=timezone,
        past_days=past_days,
        forecast_days=forecast_days,
        start_date=start_date,
        end_date=end_date,
    )

    weather_df = fetch_weather_data(
        latitude=latitude,
        longitude=longitude,
        timezone=timezone,
        past_days=past_days,
        forecast_days=forecast_days,
        start_date=start_date,
        end_date=end_date,
    )

    print(f"AQI rows for {city}: {len(aqi_df)}")
    print(f"Weather rows for {city}: {len(weather_df)}")

    if aqi_df.empty and weather_df.empty:
        return pd.DataFrame()

    if aqi_df.empty:
        merged_df = weather_df.copy()
    elif weather_df.empty:
        merged_df = aqi_df.copy()
    else:
        merge_keys = ["time", "latitude", "longitude"]
        merged_df = pd.merge(
            aqi_df,
            weather_df,
            on=merge_keys,
            how="outer",
            suffixes=("", "_weather"),
        )

    merged_df["city"] = city
    merged_df["latitude"] = round(float(latitude), 4)
    merged_df["longitude"] = round(float(longitude), 4)

    print(f"Feature rows for {city}: {len(merged_df)}")

    return merged_df


def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = [
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
        for column in df.columns
    ]

    return df


def cap_series(series: pd.Series, lower=None, upper=None) -> pd.Series:
    cleaned = pd.to_numeric(series, errors="coerce")

    if lower is not None:
        cleaned = cleaned.mask(cleaned < lower, np.nan)

    if upper is not None:
        cleaned = cleaned.mask(cleaned > upper, upper)

    return cleaned


def clean_raw_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

    df = df.dropna(subset=["time", "city"])
    df["city"] = df["city"].astype(str).str.strip()
    df = df[df["city"].str.lower() != "none"].copy()

    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "us_aqi" in df.columns:
        df["us_aqi"] = cap_series(df["us_aqi"], lower=0, upper=500)

    for col in POLLUTANT_COLUMNS:
        if col in df.columns:
            df[col] = cap_series(df[col], lower=0, upper=None)

    pollutant_caps = {
        "pm2_5": 1000,
        "pm10": 2000,
        "carbon_monoxide": 50000,
        "nitrogen_dioxide": 1000,
        "sulphur_dioxide": 1000,
        "ozone": 1000,
        "dust": 3000,
    }

    for col, upper_limit in pollutant_caps.items():
        if col in df.columns:
            df[col] = cap_series(df[col], lower=0, upper=upper_limit)

    if "temperature_2m" in df.columns:
        df["temperature_2m"] = cap_series(df["temperature_2m"], lower=-10, upper=60)

    if "relative_humidity_2m" in df.columns:
        df["relative_humidity_2m"] = cap_series(df["relative_humidity_2m"], lower=0, upper=100)

    if "precipitation" in df.columns:
        df["precipitation"] = cap_series(df["precipitation"], lower=0, upper=300)

    if "surface_pressure" in df.columns:
        df["surface_pressure"] = cap_series(df["surface_pressure"], lower=850, upper=1100)

    if "wind_speed_10m" in df.columns:
        df["wind_speed_10m"] = cap_series(df["wind_speed_10m"], lower=0, upper=160)

    if "wind_direction_10m" in df.columns:
        wind_direction = pd.to_numeric(df["wind_direction_10m"], errors="coerce")
        wind_direction = wind_direction % 360
        df["wind_direction_10m"] = wind_direction

    return df


def fill_missing_values_by_city(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["city", "time"]).reset_index(drop=True)

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    protected_columns = [
        "latitude",
        "longitude",
    ]

    fill_columns = [col for col in numeric_columns if col not in protected_columns]

    for city, city_index in df.groupby("city").groups.items():
        city_idx = list(city_index)

        df.loc[city_idx, fill_columns] = (
            df.loc[city_idx, fill_columns]
            .ffill()
            .bfill()
        )

    for col in fill_columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def remove_extreme_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    outlier_columns = [
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
        "surface_pressure",
        "wind_speed_10m",
    ]

    for col in outlier_columns:
        if col not in df.columns:
            continue

        cleaned_chunks = []

        for city, city_df in df.groupby("city"):
            city_df = city_df.copy()

            series = pd.to_numeric(city_df[col], errors="coerce")

            if series.dropna().shape[0] < 20:
                cleaned_chunks.append(city_df)
                continue

            q1 = series.quantile(0.01)
            q99 = series.quantile(0.99)

            city_df[col] = series.clip(lower=q1, upper=q99)

            cleaned_chunks.append(city_df)

        df = pd.concat(cleaned_chunks, ignore_index=True)

    df = df.sort_values(["city", "time"]).reset_index(drop=True)

    return df


def clean_feature_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Starting data cleaning...")

    initial_rows = len(df)

    df = remove_duplicate_columns(df)
    df = clean_column_names(df)

    df = df.drop_duplicates(subset=["time", "city"], keep="last")

    df = clean_raw_values(df)
    df = fill_missing_values_by_city(df)
    df = remove_extreme_outliers_iqr(df)
    df = fill_missing_values_by_city(df)

    final_rows = len(df)

    print(f"Data cleaning completed. Rows before: {initial_rows}, after: {final_rows}")

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek
    df["day_of_month"] = df["time"].dt.day
    df["month"] = df["time"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df


def add_city_encoding(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    sorted_cities = sorted(df["city"].dropna().unique())

    city_to_id = {city: index for index, city in enumerate(sorted_cities)}

    df["city_id"] = df["city"].map(city_to_id).fillna(-1).astype(int)

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["city", "time"]).reset_index(drop=True)

    for col in LAG_COLUMNS:
        if col not in df.columns:
            continue

        for lag in LAG_HOURS:
            df[f"{col}_lag_{lag}h"] = df.groupby("city")[col].shift(lag)

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["city", "time"]).reset_index(drop=True)

    for col in ROLLING_COLUMNS:
        if col not in df.columns:
            continue

        for window in ROLLING_WINDOWS:
            df[f"{col}_rolling_mean_{window}h"] = (
                df.groupby("city")[col]
                .transform(lambda series: series.shift(1).rolling(window=window, min_periods=max(2, window // 3)).mean())
            )

            df[f"{col}_rolling_std_{window}h"] = (
                df.groupby("city")[col]
                .transform(lambda series: series.shift(1).rolling(window=window, min_periods=max(2, window // 3)).std())
            )

    return df


def add_pollutant_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "pm2_5" in df.columns and "pm10" in df.columns:
        df["pm25_to_pm10_ratio"] = df["pm2_5"] / df["pm10"].replace(0, np.nan)
        df["pm25_to_pm10_ratio"] = df["pm25_to_pm10_ratio"].replace([np.inf, -np.inf], np.nan)

    if "us_aqi" in df.columns and "pm2_5" in df.columns:
        df["aqi_to_pm25_ratio"] = df["us_aqi"] / df["pm2_5"].replace(0, np.nan)
        df["aqi_to_pm25_ratio"] = df["aqi_to_pm25_ratio"].replace([np.inf, -np.inf], np.nan)

    if "us_aqi" in df.columns and "pm10" in df.columns:
        df["aqi_to_pm10_ratio"] = df["us_aqi"] / df["pm10"].replace(0, np.nan)
        df["aqi_to_pm10_ratio"] = df["aqi_to_pm10_ratio"].replace([np.inf, -np.inf], np.nan)

    return df


def add_weather_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "temperature_2m" in df.columns and "relative_humidity_2m" in df.columns:
        df["temp_humidity_interaction"] = df["temperature_2m"] * df["relative_humidity_2m"]

    if "wind_speed_10m" in df.columns and "pm2_5" in df.columns:
        df["wind_pm25_interaction"] = df["wind_speed_10m"] * df["pm2_5"]

    if "wind_speed_10m" in df.columns and "pm10" in df.columns:
        df["wind_pm10_interaction"] = df["wind_speed_10m"] * df["pm10"]

    if "relative_humidity_2m" in df.columns and "pm2_5" in df.columns:
        df["humidity_pm25_interaction"] = df["relative_humidity_2m"] * df["pm2_5"]

    if "surface_pressure" in df.columns and "wind_speed_10m" in df.columns:
        df["pressure_wind_interaction"] = df["surface_pressure"] * df["wind_speed_10m"]

    return df


def add_future_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["city", "time"]).reset_index(drop=True)

    future_weather_columns = {
        "temperature_2m": "temp",
        "relative_humidity_2m": "humidity",
        "surface_pressure": "pressure",
        "wind_speed_10m": "wind",
        "precipitation": "precip",
    }

    day_windows = {
        "day1": (1, 24),
        "day2": (25, 48),
        "day3": (49, 72),
    }

    for source_col, short_name in future_weather_columns.items():
        if source_col not in df.columns:
            continue

        for day_name, (start_hour, end_hour) in day_windows.items():
            future_values = []

            for _, city_df in df.groupby("city"):
                city_series = city_df[source_col].reset_index(drop=True)

                city_future_values = []

                for index in range(len(city_series)):
                    window = city_series.shift(-start_hour).iloc[index:index + (end_hour - start_hour + 1)]

                    if window.dropna().empty:
                        city_future_values.append(np.nan)
                    elif source_col == "precipitation":
                        city_future_values.append(window.sum())
                    else:
                        city_future_values.append(window.mean())

                future_values.extend(city_future_values)

            feature_name = f"future_{short_name}_{day_name}_avg"

            if source_col == "precipitation":
                feature_name = f"future_{short_name}_{day_name}_sum"

            df[feature_name] = future_values

    if "wind_speed_10m" in df.columns:
        for day_name, (start_hour, end_hour) in day_windows.items():
            future_values = []

            for _, city_df in df.groupby("city"):
                city_series = city_df["wind_speed_10m"].reset_index(drop=True)

                city_future_values = []

                for index in range(len(city_series)):
                    window = city_series.shift(-start_hour).iloc[index:index + (end_hour - start_hour + 1)]

                    if window.dropna().empty:
                        city_future_values.append(np.nan)
                    else:
                        city_future_values.append(window.max())

                future_values.extend(city_future_values)

            df[f"future_wind_{day_name}_max"] = future_values

    return df


def add_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["city", "time"]).reset_index(drop=True)

    target_windows = {
        "target_aqi_day1_avg": (1, 24),
        "target_aqi_day2_avg": (25, 48),
        "target_aqi_day3_avg": (49, 72),
        "target_aqi_next_72h_avg": (1, 72),
    }

    if "us_aqi" not in df.columns:
        return df

    for target_col, (start_hour, end_hour) in target_windows.items():
        target_values = []

        for _, city_df in df.groupby("city"):
            city_series = city_df["us_aqi"].reset_index(drop=True)

            city_target_values = []

            for index in range(len(city_series)):
                future_window = city_series.shift(-start_hour).iloc[index:index + (end_hour - start_hour + 1)]

                if future_window.dropna().empty:
                    city_target_values.append(np.nan)
                else:
                    city_target_values.append(future_window.mean())

            target_values.extend(city_target_values)

        df[target_col] = target_values

    return df


def clean_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.replace([np.inf, -np.inf], np.nan)

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for city, city_index in df.groupby("city").groups.items():
        city_idx = list(city_index)

        df.loc[city_idx, numeric_columns] = (
            df.loc[city_idx, numeric_columns]
            .ffill()
            .bfill()
        )

    for col in numeric_columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    df = df.drop_duplicates(subset=["time", "city"], keep="last")
    df = df.sort_values(["city", "time"]).reset_index(drop=True)

    return df


def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    print("Building features...")

    df = raw_df.copy()

    df = clean_feature_data(df)
    df = add_time_features(df)
    df = add_city_encoding(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_pollutant_ratios(df)
    df = add_weather_interaction_features(df)
    df = add_future_weather_features(df)
    df = add_target_columns(df)
    df = clean_engineered_features(df)

    print(f"Feature engineering completed. Final shape: {df.shape}")

    return df


def fetch_all_cities(args) -> pd.DataFrame:
    city_frames = []

    if args.latitude is not None and args.longitude is not None:
        custom_city_name = "Custom Location"

        custom_df = fetch_city_data(
            city=custom_city_name,
            latitude=args.latitude,
            longitude=args.longitude,
            timezone=args.timezone,
            past_days=args.past_days,
            forecast_days=args.forecast_days,
            start_date=args.start_date,
            end_date=args.end_date,
        )

        if not custom_df.empty:
            city_frames.append(custom_df)

    else:
        for city, location in CITY_LOCATIONS.items():
            city_df = fetch_city_data(
                city=city,
                latitude=location["latitude"],
                longitude=location["longitude"],
                timezone=args.timezone,
                past_days=args.past_days,
                forecast_days=args.forecast_days,
                start_date=args.start_date,
                end_date=args.end_date,
            )

            if not city_df.empty:
                city_frames.append(city_df)

    if not city_frames:
        return pd.DataFrame()

    combined_df = pd.concat(city_frames, ignore_index=True)

    return combined_df


def merge_with_existing_feature_store(new_features: pd.DataFrame) -> pd.DataFrame:
    if not FEATURE_STORE_PATH.exists():
        return new_features

    try:
        existing_df = pd.read_parquet(FEATURE_STORE_PATH)

        if existing_df.empty:
            return new_features

        existing_df = clean_column_names(existing_df)
        new_features = clean_column_names(new_features)

        combined_df = pd.concat([existing_df, new_features], ignore_index=True)

        if "time" in combined_df.columns:
            combined_df["time"] = pd.to_datetime(combined_df["time"], errors="coerce")

        combined_df = combined_df.dropna(subset=["time", "city"])
        combined_df = combined_df.drop_duplicates(subset=["time", "city"], keep="last")
        combined_df = combined_df.sort_values(["city", "time"]).reset_index(drop=True)

        return combined_df

    except Exception as error:
        print("Could not merge with existing feature store. Using new features only.")
        print(error)
        return new_features


def save_local_feature_store(features: pd.DataFrame):
    FEATURE_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

    features.to_parquet(FEATURE_STORE_PATH, index=False)

    print(f"Saved local feature store: {FEATURE_STORE_PATH}")
    print(f"Local feature store rows: {len(features)}")


def main():
    args = parse_args()

    print("Starting feature pipeline...")
    print(f"Timezone: {args.timezone}")
    print(f"Past days: {args.past_days}")
    print(f"Forecast days: {args.forecast_days}")

    raw_df = fetch_all_cities(args)

    if raw_df.empty:
        raise RuntimeError("No data fetched from APIs.")

    print(f"Raw fetched rows: {len(raw_df)}")

    new_features = build_features(raw_df)

    merged_features = merge_with_existing_feature_store(new_features)

    merged_features = clean_feature_data(merged_features)
    merged_features = add_time_features(merged_features)
    merged_features = add_city_encoding(merged_features)
    merged_features = add_lag_features(merged_features)
    merged_features = add_rolling_features(merged_features)
    merged_features = add_pollutant_ratios(merged_features)
    merged_features = add_weather_interaction_features(merged_features)
    merged_features = add_future_weather_features(merged_features)
    merged_features = add_target_columns(merged_features)
    merged_features = clean_engineered_features(merged_features)

    save_local_feature_store(merged_features)

    if args.skip_hopsworks:
        print("Hopsworks upload skipped by argument.")
    else:
        cloud_saved = save_features_to_hopsworks(merged_features)

        if cloud_saved:
            print("Cloud feature store status: Hopsworks upload successful.")
        else:
            print("Cloud feature store status: Hopsworks upload skipped or failed. Local backup is still saved.")

    print("Feature pipeline completed.")


if __name__ == "__main__":
    main()