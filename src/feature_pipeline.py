import argparse
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from config import (
    FEATURE_STORE_PATH,
    UAE_LOCATIONS,
    TARGET_COLUMNS,
)

from hopsworks_store import save_features_to_hopsworks


AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


AQI_HOURLY_VARIABLES = [
    "us_aqi",
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "dust",
]


WEATHER_HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
]


def request_json(url: str, params: dict, max_retries: int = 3) -> dict:
    last_error = None

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=60)

            if response.status_code in [429, 500, 502, 503, 504]:
                print(
                    f"Temporary API error {response.status_code}. "
                    f"Attempt {attempt + 1}/{max_retries}. Retrying..."
                )
                time.sleep(2 + attempt * 3)
                continue

            response.raise_for_status()
            return response.json()

        except Exception as error:
            last_error = error
            print(f"API request failed. Attempt {attempt + 1}/{max_retries}. Error: {error}")
            time.sleep(2 + attempt * 3)

    raise RuntimeError(f"API request failed after {max_retries} attempts: {last_error}")


def fetch_air_quality(
    latitude: float,
    longitude: float,
    timezone: str,
    start_date: str | None = None,
    end_date: str | None = None,
    past_days: int | None = None,
    forecast_days: int = 4,
) -> pd.DataFrame:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone,
        "hourly": ",".join(AQI_HOURLY_VARIABLES),
        "forecast_days": forecast_days,
    }

    if start_date and end_date:
        params["start_date"] = start_date
        params["end_date"] = end_date
    elif past_days is not None:
        params["past_days"] = past_days

    payload = request_json(AIR_QUALITY_URL, params=params)
    hourly = payload.get("hourly", {})

    if not hourly or "time" not in hourly:
        raise ValueError(f"Unexpected AQI API response: {payload}")

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df["latitude"] = float(latitude)
    df["longitude"] = float(longitude)

    return df.sort_values("time").reset_index(drop=True)


def fetch_weather(
    latitude: float,
    longitude: float,
    timezone: str,
    start_date: str | None = None,
    end_date: str | None = None,
    past_days: int | None = None,
    forecast_days: int = 4,
) -> pd.DataFrame:
    use_archive = start_date is not None and end_date is not None

    if use_archive:
        url = WEATHER_ARCHIVE_URL
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "timezone": timezone,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(WEATHER_HOURLY_VARIABLES),
        }
    else:
        url = WEATHER_FORECAST_URL
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "timezone": timezone,
            "past_days": past_days if past_days is not None else 92,
            "forecast_days": forecast_days,
            "hourly": ",".join(WEATHER_HOURLY_VARIABLES),
        }

    payload = request_json(url, params=params)
    hourly = payload.get("hourly", {})

    if not hourly or "time" not in hourly:
        raise ValueError(f"Unexpected weather API response: {payload}")

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df["latitude"] = float(latitude)
    df["longitude"] = float(longitude)

    return df.sort_values("time").reset_index(drop=True)


def forward_rolling_mean(series: pd.Series, start_hour: int, window_hours: int, min_periods: int = 12) -> pd.Series:
    future_series = series.shift(-start_hour)
    return (
        future_series
        .iloc[::-1]
        .rolling(window=window_hours, min_periods=min_periods)
        .mean()
        .iloc[::-1]
    )


def forward_rolling_max(series: pd.Series, start_hour: int, window_hours: int, min_periods: int = 12) -> pd.Series:
    future_series = series.shift(-start_hour)
    return (
        future_series
        .iloc[::-1]
        .rolling(window=window_hours, min_periods=min_periods)
        .max()
        .iloc[::-1]
    )


def forward_rolling_sum(series: pd.Series, start_hour: int, window_hours: int, min_periods: int = 12) -> pd.Series:
    future_series = series.shift(-start_hour)
    return (
        future_series
        .iloc[::-1]
        .rolling(window=window_hours, min_periods=min_periods)
        .sum()
        .iloc[::-1]
    )


def merge_air_quality_and_weather(aqi_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    merge_columns = ["time", "latitude", "longitude"]

    df = pd.merge(
        aqi_df,
        weather_df,
        on=merge_columns,
        how="left",
    )

    df = df.sort_values("time").reset_index(drop=True)

    return df


def make_features_for_location(df: pd.DataFrame, city: str) -> pd.DataFrame:
    df = df.copy().sort_values("time").reset_index(drop=True)

    df["city"] = city

    df["location_key"] = (
        df["latitude"].round(4).astype(str)
        + "_"
        + df["longitude"].round(4).astype(str)
    )

    df["hour"] = df["time"].dt.hour
    df["day"] = df["time"].dt.day
    df["month"] = df["time"].dt.month
    df["dayofweek"] = df["time"].dt.dayofweek

    df["aqi_lag_1h"] = df["us_aqi"].shift(1)
    df["aqi_lag_3h"] = df["us_aqi"].shift(3)
    df["aqi_lag_6h"] = df["us_aqi"].shift(6)
    df["aqi_lag_12h"] = df["us_aqi"].shift(12)
    df["aqi_lag_24h"] = df["us_aqi"].shift(24)

    df["pm25_lag_24h"] = df["pm2_5"].shift(24)
    df["pm10_lag_24h"] = df["pm10"].shift(24)

    df["aqi_rolling_mean_3h"] = df["us_aqi"].rolling(3, min_periods=2).mean()
    df["aqi_rolling_mean_6h"] = df["us_aqi"].rolling(6, min_periods=3).mean()
    df["aqi_rolling_mean_12h"] = df["us_aqi"].rolling(12, min_periods=6).mean()
    df["aqi_rolling_mean_24h"] = df["us_aqi"].rolling(24, min_periods=12).mean()

    df["pm25_rolling_mean_24h"] = df["pm2_5"].rolling(24, min_periods=12).mean()
    df["pm10_rolling_mean_24h"] = df["pm10"].rolling(24, min_periods=12).mean()
    df["ozone_rolling_mean_24h"] = df["ozone"].rolling(24, min_periods=12).mean()
    df["wind_speed_rolling_mean_24h"] = df["wind_speed_10m"].rolling(24, min_periods=12).mean()
    df["pressure_rolling_mean_24h"] = df["surface_pressure"].rolling(24, min_periods=12).mean()

    df["aqi_change_rate_1h"] = df["us_aqi"].diff(1)
    df["aqi_change_rate_3h"] = df["us_aqi"].diff(3)
    df["aqi_change_rate_24h"] = df["us_aqi"].diff(24)

    df["future_temp_day1_avg"] = forward_rolling_mean(df["temperature_2m"], 1, 24)
    df["future_temp_day2_avg"] = forward_rolling_mean(df["temperature_2m"], 25, 24)
    df["future_temp_day3_avg"] = forward_rolling_mean(df["temperature_2m"], 49, 24)

    df["future_humidity_day1_avg"] = forward_rolling_mean(df["relative_humidity_2m"], 1, 24)
    df["future_humidity_day2_avg"] = forward_rolling_mean(df["relative_humidity_2m"], 25, 24)
    df["future_humidity_day3_avg"] = forward_rolling_mean(df["relative_humidity_2m"], 49, 24)

    df["future_wind_day1_avg"] = forward_rolling_mean(df["wind_speed_10m"], 1, 24)
    df["future_wind_day2_avg"] = forward_rolling_mean(df["wind_speed_10m"], 25, 24)
    df["future_wind_day3_avg"] = forward_rolling_mean(df["wind_speed_10m"], 49, 24)

    df["future_wind_day1_max"] = forward_rolling_max(df["wind_speed_10m"], 1, 24)
    df["future_wind_day2_max"] = forward_rolling_max(df["wind_speed_10m"], 25, 24)
    df["future_wind_day3_max"] = forward_rolling_max(df["wind_speed_10m"], 49, 24)

    df["future_wind_direction_day1_avg"] = forward_rolling_mean(df["wind_direction_10m"], 1, 24)
    df["future_wind_direction_day2_avg"] = forward_rolling_mean(df["wind_direction_10m"], 25, 24)
    df["future_wind_direction_day3_avg"] = forward_rolling_mean(df["wind_direction_10m"], 49, 24)

    df["future_pressure_day1_avg"] = forward_rolling_mean(df["surface_pressure"], 1, 24)
    df["future_pressure_day2_avg"] = forward_rolling_mean(df["surface_pressure"], 25, 24)
    df["future_pressure_day3_avg"] = forward_rolling_mean(df["surface_pressure"], 49, 24)

    df["future_precip_day1_sum"] = forward_rolling_sum(df["precipitation"], 1, 24)
    df["future_precip_day2_sum"] = forward_rolling_sum(df["precipitation"], 25, 24)
    df["future_precip_day3_sum"] = forward_rolling_sum(df["precipitation"], 49, 24)

    df[TARGET_COLUMNS["day_1"]] = forward_rolling_mean(df["us_aqi"], 1, 24)
    df[TARGET_COLUMNS["day_2"]] = forward_rolling_mean(df["us_aqi"], 25, 24)
    df[TARGET_COLUMNS["day_3"]] = forward_rolling_mean(df["us_aqi"], 49, 24)
    df[TARGET_COLUMNS["next_72h_avg"]] = forward_rolling_mean(df["us_aqi"], 1, 72, min_periods=36)

    df = df.replace([float("inf"), float("-inf")], pd.NA)

    return df


def fetch_location_dataset(
    city: str,
    latitude: float,
    longitude: float,
    timezone: str,
    start_date: str | None,
    end_date: str | None,
    past_days: int,
    forecast_days: int,
) -> pd.DataFrame:
    print(f"Fetching data for {city}...")

    aqi_df = fetch_air_quality(
        latitude=latitude,
        longitude=longitude,
        timezone=timezone,
        start_date=start_date,
        end_date=end_date,
        past_days=past_days if not (start_date and end_date) else None,
        forecast_days=forecast_days,
    )

    weather_df = fetch_weather(
        latitude=latitude,
        longitude=longitude,
        timezone=timezone,
        start_date=start_date,
        end_date=end_date,
        past_days=past_days if not (start_date and end_date) else None,
        forecast_days=forecast_days,
    )

    print(f"AQI rows for {city}: {len(aqi_df)}")
    print(f"Weather rows for {city}: {len(weather_df)}")

    combined_df = merge_air_quality_and_weather(aqi_df, weather_df)
    feature_df = make_features_for_location(combined_df, city=city)

    print(f"Feature rows for {city}: {len(feature_df)}")

    return feature_df


def save_feature_store(features: pd.DataFrame) -> pd.DataFrame:
    FEATURE_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

    features = features.copy()
    features["time"] = pd.to_datetime(features["time"])

    if FEATURE_STORE_PATH.exists():
        old_df = pd.read_parquet(FEATURE_STORE_PATH)
        old_df["time"] = pd.to_datetime(old_df["time"])

        merged_df = pd.concat([old_df, features], ignore_index=True)
    else:
        merged_df = features

    merged_df["location_key"] = (
        merged_df["latitude"].round(4).astype(str)
        + "_"
        + merged_df["longitude"].round(4).astype(str)
    )

    merged_df = merged_df.drop_duplicates(
        subset=["time", "latitude", "longitude"],
        keep="last",
    )

    merged_df = merged_df.sort_values(["city", "time"]).reset_index(drop=True)
    merged_df.to_parquet(FEATURE_STORE_PATH, index=False)

    print(f"Saved local feature store: {FEATURE_STORE_PATH}")
    print(f"Local feature store rows: {len(merged_df)}")

    return merged_df


def build_all_features(
    timezone: str,
    start_date: str | None,
    end_date: str | None,
    past_days: int,
    forecast_days: int,
) -> pd.DataFrame:
    all_location_features = []

    for location in UAE_LOCATIONS:
        try:
            location_df = fetch_location_dataset(
                city=location["city"],
                latitude=location["latitude"],
                longitude=location["longitude"],
                timezone=timezone,
                start_date=start_date,
                end_date=end_date,
                past_days=past_days,
                forecast_days=forecast_days,
            )

            all_location_features.append(location_df)

        except Exception as error:
            print(f"Failed to process {location['city']}: {error}")

    if not all_location_features:
        raise RuntimeError("No feature data was created for any location.")

    combined_features = pd.concat(all_location_features, ignore_index=True)
    combined_features = combined_features.sort_values(["city", "time"]).reset_index(drop=True)

    return combined_features


def main():
    parser = argparse.ArgumentParser(
        description="Fetch AQI/weather data, engineer features, save locally and optionally to Hopsworks Feature Store."
    )

    parser.add_argument("--timezone", type=str, default="Asia/Dubai")
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--past-days", type=int, default=92)
    parser.add_argument("--forecast-days", type=int, default=4)
    parser.add_argument("--skip-hopsworks", action="store_true")

    args = parser.parse_args()

    print("Starting AQI feature pipeline...")
    print(f"Timezone: {args.timezone}")

    if args.start_date and args.end_date:
        print(f"Mode: historical backfill from {args.start_date} to {args.end_date}")
    else:
        print(f"Mode: recent data with past_days={args.past_days}, forecast_days={args.forecast_days}")

    features = build_all_features(
        timezone=args.timezone,
        start_date=args.start_date,
        end_date=args.end_date,
        past_days=args.past_days,
        forecast_days=args.forecast_days,
    )

    merged_features = save_feature_store(features)

    if args.skip_hopsworks:
        print("Skipping Hopsworks upload because --skip-hopsworks was provided.")
    else:
        cloud_saved = save_features_to_hopsworks(merged_features)

        if cloud_saved:
            print("Cloud feature store status: Hopsworks upload successful.")
        else:
            print("Cloud feature store status: Hopsworks upload skipped or failed. Local backup is still saved.")

    print("Feature pipeline completed.")


if __name__ == "__main__":
    main()