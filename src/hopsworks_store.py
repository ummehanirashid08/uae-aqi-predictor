from __future__ import annotations

import json
import os
import shutil
import time
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import joblib
import pandas as pd
from dotenv import load_dotenv

from config import get_bool_setting, get_int_setting, get_setting


load_dotenv(override=True)


def _safe_int_env(name: str, default: int) -> int:
    return get_int_setting(name, default)


# Important:
# Old feature group name "aqi_features" has broken/partial versions in your Hopsworks project.
# So this file forces a clean feature group unless you intentionally set a different clean name.
_RAW_FEATURE_GROUP_NAME = str(get_setting("FEATURE_GROUP_NAME", "aqi_features_clean")).strip()

if not _RAW_FEATURE_GROUP_NAME or _RAW_FEATURE_GROUP_NAME == "aqi_features":
    FEATURE_GROUP_NAME = "aqi_features_clean"
else:
    FEATURE_GROUP_NAME = _RAW_FEATURE_GROUP_NAME

FEATURE_GROUP_VERSION = _safe_int_env("FEATURE_GROUP_VERSION", 2)
HOPSWORKS_UPLOAD_CHUNK_SIZE = _safe_int_env("HOPSWORKS_UPLOAD_CHUNK_SIZE", 3500)

MODEL_NAME = str(get_setting("MODEL_NAME", "uae_aqi_forecast_model")).strip()
MODEL_VERSION = _safe_int_env("MODEL_VERSION", 1)

PROJECT_NAME = str(get_setting("HOPSWORKS_PROJECT_NAME", "")).strip()
API_KEY = str(get_setting("HOPSWORKS_API_KEY", "")).strip()
USE_HOPSWORKS = get_bool_setting("USE_HOPSWORKS", False)


def print_hopsworks_config() -> None:
    print("Hopsworks config check:")
    print(f"USE_HOPSWORKS={USE_HOPSWORKS}")
    print(f"HOPSWORKS_PROJECT_NAME={PROJECT_NAME}")
    print(f"FEATURE_GROUP_NAME={FEATURE_GROUP_NAME}")
    print(f"FEATURE_GROUP_VERSION={FEATURE_GROUP_VERSION}")
    print(f"HOPSWORKS_UPLOAD_CHUNK_SIZE={HOPSWORKS_UPLOAD_CHUNK_SIZE}")
    print(f"MODEL_NAME={MODEL_NAME}")
    print(f"MODEL_VERSION={MODEL_VERSION}")
    print(f"API_KEY_EXISTS={bool(API_KEY)}")


def hopsworks_enabled() -> bool:
    if not USE_HOPSWORKS:
        print("Hopsworks disabled. Set USE_HOPSWORKS=true to enable it.")
        return False

    if not PROJECT_NAME:
        print("Hopsworks project name missing. Set HOPSWORKS_PROJECT_NAME in .env.")
        return False

    if not API_KEY:
        print("Hopsworks API key missing. Set HOPSWORKS_API_KEY in .env.")
        return False

    return True


def get_hopsworks_project():
    print_hopsworks_config()

    if not hopsworks_enabled():
        return None

    try:
        import hopsworks

        project = hopsworks.login(
            project=PROJECT_NAME,
            api_key_value=API_KEY,
        )

        print(f"Connected to Hopsworks project: {PROJECT_NAME}")
        return project

    except Exception as error:
        print(f"Could not connect to Hopsworks: {error}")
        return None


def get_feature_store():
    project = get_hopsworks_project()

    if project is None:
        return None

    try:
        feature_store = project.get_feature_store()
        print("Connected to Hopsworks Feature Store.")
        return feature_store

    except Exception as error:
        print(f"Could not connect to Hopsworks Feature Store: {error}")
        return None


def _normalize_column_name(column: str) -> str:
    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def _clean_column_values(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = df.copy()

    for column in clean_df.columns:
        if column == "time":
            continue

        if pd.api.types.is_object_dtype(clean_df[column]):
            clean_df[column] = clean_df[column].astype(str)
            clean_df[column] = clean_df[column].replace(
                ["nan", "NaN", "None", "none", "NULL", "null", "<NA>"],
                "",
            )

        elif pd.api.types.is_float_dtype(clean_df[column]):
            clean_df[column] = pd.to_numeric(
                clean_df[column],
                errors="coerce",
            ).astype("float64")

        elif pd.api.types.is_integer_dtype(clean_df[column]):
            clean_df[column] = (
                pd.to_numeric(clean_df[column], errors="coerce")
                .fillna(0)
                .astype("int64")
            )

    return clean_df


def prepare_features_for_hopsworks(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Cannot prepare empty dataframe for Hopsworks.")

    prepared_df = df.copy()
    prepared_df.columns = [_normalize_column_name(column) for column in prepared_df.columns]

    if "time" not in prepared_df.columns:
        raise ValueError("Feature dataframe must contain a time column.")

    prepared_df["time"] = pd.to_datetime(prepared_df["time"], errors="coerce")
    prepared_df = prepared_df.dropna(subset=["time"]).copy()

    if "location_key" not in prepared_df.columns:
        if "latitude" in prepared_df.columns and "longitude" in prepared_df.columns:
            lat = pd.to_numeric(prepared_df["latitude"], errors="coerce").round(4).astype(str)
            lon = pd.to_numeric(prepared_df["longitude"], errors="coerce").round(4).astype(str)
            prepared_df["location_key"] = lat + "_" + lon

        elif "city" in prepared_df.columns:
            prepared_df["location_key"] = prepared_df["city"].astype(str)

        else:
            prepared_df["location_key"] = "unknown_location"

    prepared_df["location_key"] = prepared_df["location_key"].astype(str)
    prepared_df["location_key"] = prepared_df["location_key"].replace(
        ["", "nan", "NaN", "None", "none", "NULL", "null", "<NA>"],
        pd.NA,
    )

    if "city" in prepared_df.columns:
        prepared_df["location_key"] = prepared_df["location_key"].fillna(
            prepared_df["city"].astype(str)
        )

    prepared_df["location_key"] = prepared_df["location_key"].fillna("unknown_location")
    prepared_df = prepared_df[prepared_df["location_key"].notna()].copy()

    if "city" in prepared_df.columns:
        prepared_df["city"] = prepared_df["city"].astype(str).str.strip()

    prepared_df = prepared_df.sort_values(["location_key", "time"]).reset_index(drop=True)
    prepared_df = _clean_column_values(prepared_df)

    print(f"Hopsworks prepared dataframe rows: {len(prepared_df)}")
    print(f"Hopsworks prepared dataframe columns: {len(prepared_df.columns)}")
    print(f"Hopsworks location_key null count: {prepared_df['location_key'].isna().sum()}")
    print(f"Hopsworks time null count: {prepared_df['time'].isna().sum()}")

    return prepared_df


def _get_existing_feature_group(feature_store):
    try:
        feature_group = feature_store.get_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
        )

        if feature_group is None:
            print(
                f"Hopsworks returned None for existing feature group "
                f"{FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION}."
            )
            return None

        print(
            f"Using existing Hopsworks feature group: "
            f"{FEATURE_GROUP_NAME}, version {FEATURE_GROUP_VERSION}"
        )
        return feature_group

    except Exception as error:
        print(
            f"Existing feature group not found or not readable: "
            f"{FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION}. Reason: {error}"
        )
        return None


def _create_clean_batch_feature_group(feature_store):
    print(
        f"Creating clean batch Hopsworks feature group: "
        f"{FEATURE_GROUP_NAME}, version {FEATURE_GROUP_VERSION}"
    )

    try:
        feature_group = feature_store.create_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
            description=(
                "Batch-written UAE AQI feature group with pollutant, weather, lag, "
                "rolling, interaction, and future forecast features."
            ),
            primary_key=["location_key", "time"],
            event_time="time",
            online_enabled=False,
            stream=False,
        )

    except TypeError:
        feature_group = feature_store.create_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
            description=(
                "Batch-written UAE AQI feature group with pollutant, weather, lag, "
                "rolling, interaction, and future forecast features."
            ),
            primary_key=["location_key", "time"],
            event_time="time",
            online_enabled=False,
        )

    if feature_group is None:
        raise RuntimeError(
            f"Hopsworks returned None while creating feature group "
            f"{FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION}."
        )

    return feature_group


def _get_or_create_batch_feature_group(feature_store):
    feature_group = _get_existing_feature_group(feature_store)

    if feature_group is not None:
        return feature_group

    return _create_clean_batch_feature_group(feature_store)


def _insert_feature_chunk(feature_group, chunk_df: pd.DataFrame) -> None:
    feature_group.insert(
        chunk_df,
        write_options={"wait_for_job": True},
    )


def _upload_feature_chunks(feature_group, prepared_df: pd.DataFrame) -> None:
    chunk_size = max(1, HOPSWORKS_UPLOAD_CHUNK_SIZE)
    total_rows = len(prepared_df)
    total_chunks = (total_rows + chunk_size - 1) // chunk_size

    print(
        f"Uploading features to Hopsworks in {total_chunks} chunks "
        f"of up to {chunk_size} rows."
    )
    print("Each chunk uses wait_for_job=True.")

    for chunk_index, start_index in enumerate(range(0, total_rows, chunk_size), start=1):
        end_index = min(start_index + chunk_size, total_rows)
        chunk_df = prepared_df.iloc[start_index:end_index].copy()

        print(
            f"Uploading Hopsworks chunk {chunk_index}/{total_chunks}: "
            f"rows {start_index + 1}-{end_index} of {total_rows}"
        )

        for attempt in range(1, 4):
            try:
                print(f"Chunk {chunk_index}/{total_chunks}, attempt {attempt}/3")
                _insert_feature_chunk(feature_group, chunk_df)
                print(f"Chunk {chunk_index}/{total_chunks} uploaded successfully.")
                break

            except Exception as error:
                print(
                    f"Chunk {chunk_index}/{total_chunks} failed on attempt "
                    f"{attempt}/3: {error}"
                )

                if attempt == 3:
                    raise

                print("Waiting 20 seconds before retrying this chunk...")
                time.sleep(20)


def save_features_to_hopsworks(df: pd.DataFrame) -> bool:
    feature_store = get_feature_store()

    if feature_store is None:
        print("Hopsworks Feature Store unavailable. Skipping cloud upload.")
        return False

    try:
        prepared_df = prepare_features_for_hopsworks(df)

        print(
            f"Prepared dataframe for Hopsworks batch upload: "
            f"{len(prepared_df)} rows, {len(prepared_df.columns)} columns"
        )

        feature_group = _get_or_create_batch_feature_group(feature_store)

        print("Uploading features to Hopsworks with chunked wait_for_job=True inserts.")
        print("This can take several minutes. Do not stop the terminal.")

        _upload_feature_chunks(feature_group, prepared_df)

        print("Hopsworks upload successful.")
        print("Cloud feature store status: Hopsworks upload successful.")
        return True

    except Exception as error:
        print(f"Hopsworks feature save failed: {error}")
        print(
            "Cloud feature store status: Hopsworks upload skipped or failed. "
            "Local backup is still saved."
        )
        return False


def _clean_loaded_hopsworks_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_df = df.copy()
    cleaned_df.columns = [_normalize_column_name(column) for column in cleaned_df.columns]

    if "time" in cleaned_df.columns:
        cleaned_df["time"] = pd.to_datetime(cleaned_df["time"], errors="coerce")
        cleaned_df = cleaned_df.dropna(subset=["time"]).copy()

    if "location_key" in cleaned_df.columns:
        cleaned_df["location_key"] = cleaned_df["location_key"].astype(str)

    sort_columns = [
        column for column in ["location_key", "city", "time"]
        if column in cleaned_df.columns
    ]

    if sort_columns:
        cleaned_df = cleaned_df.sort_values(sort_columns).reset_index(drop=True)
    else:
        cleaned_df = cleaned_df.reset_index(drop=True)

    return cleaned_df


def _read_feature_group(feature_group) -> Tuple[Optional[pd.DataFrame], str]:
    read_errors = []

    try:
        print("Trying Hopsworks read method 1: feature_group.read()")
        df = feature_group.read()

        if df is not None and not df.empty:
            print(f"Hopsworks feature_group.read() successful: {len(df)} rows")
            return _clean_loaded_hopsworks_dataframe(df), "feature_group.read()"

        read_errors.append("feature_group.read() returned no rows")

    except Exception as error:
        message = f"feature_group.read() failed: {error}"
        read_errors.append(message)
        print(f"Hopsworks {message}")

    try:
        print("Trying Hopsworks read method 2: select_all().read()")
        query = feature_group.select_all()
        df = query.read()

        if df is not None and not df.empty:
            print(f"Hopsworks select_all().read() successful: {len(df)} rows")
            return _clean_loaded_hopsworks_dataframe(df), "select_all().read()"

        read_errors.append("select_all().read() returned no rows")

    except Exception as error:
        message = f"select_all().read() failed: {error}"
        read_errors.append(message)
        print(f"Hopsworks {message}")

    return None, "; ".join(read_errors) if read_errors else "Hopsworks read returned no data"


def load_features_from_hopsworks() -> Tuple[Optional[pd.DataFrame], str]:
    feature_store = get_feature_store()

    if feature_store is None:
        print("Hopsworks read failed, falling back to local parquet.")
        return None, "Hopsworks Feature Store unavailable; using local feature store fallback."

    try:
        print(
            f"Loading data from Hopsworks feature group: "
            f"{FEATURE_GROUP_NAME}, version {FEATURE_GROUP_VERSION}"
        )

        feature_group = feature_store.get_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
        )

        if feature_group is None:
            raise RuntimeError(
                f"Hopsworks returned None for feature group object "
                f"{FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION}."
            )

        df, read_source = _read_feature_group(feature_group)

        if df is None or df.empty:
            reason = (
                f"Hopsworks feature group {FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION} "
                f"did not return usable data. Reason: {read_source}"
            )
            print(reason)
            print("Hopsworks read failed, falling back to local parquet.")
            return None, reason

        print(f"Training data loaded from Hopsworks Feature Store: {len(df)} rows")
        return df, "Hopsworks Feature Store"

    except Exception as error:
        reason = (
            f"Could not load Hopsworks feature group {FEATURE_GROUP_NAME} "
            f"v{FEATURE_GROUP_VERSION}: {error}"
        )
        print(reason)
        print("Hopsworks read failed, falling back to local parquet.")
        return None, reason


def _flatten_metrics_for_hopsworks(metrics: dict) -> dict:
    if not isinstance(metrics, dict):
        return {}

    flat_metrics = {}

    if "best_metrics" in metrics and isinstance(metrics["best_metrics"], dict):
        for key, value in metrics["best_metrics"].items():
            if isinstance(value, (int, float)):
                flat_metrics[key] = float(value)

    if "targets" in metrics and isinstance(metrics["targets"], dict):
        for target_name, target_data in metrics["targets"].items():
            if not isinstance(target_data, dict):
                continue

            best_metrics = target_data.get("best_metrics", {})
            if isinstance(best_metrics, dict):
                for metric_name, metric_value in best_metrics.items():
                    if isinstance(metric_value, (int, float)):
                        flat_metrics[f"{target_name}_{metric_name}"] = float(metric_value)

            best_model = target_data.get("best_model")
            if best_model:
                flat_metrics[f"{target_name}_best_model"] = str(best_model)

    for key in ["rmse", "mae", "r2"]:
        if key in metrics and isinstance(metrics[key], (int, float)):
            flat_metrics[key] = float(metrics[key])

    return flat_metrics


def save_model_to_hopsworks_registry(
    model_path: Path,
    metrics: Optional[dict] = None,
    metrics_path: Optional[Path] = None,
    description: str = "Multi-horizon UAE AQI forecasting model bundle.",
) -> bool:
    if not hopsworks_enabled():
        print("Hopsworks Model Registry unavailable. Skipping registry save.")
        return False

    if not model_path.exists():
        print(f"Model path not found: {model_path}")
        return False

    project = get_hopsworks_project()

    if project is None:
        return False

    temp_dir_path = None

    try:
        model_registry = project.get_model_registry()

        temp_dir_path = Path(tempfile.mkdtemp(prefix="aqi_model_registry_"))

        temp_model_path = temp_dir_path / model_path.name
        shutil.copy2(model_path, temp_model_path)

        raw_metrics = metrics or {}

        if metrics_path is not None and Path(metrics_path).exists():
            temp_metrics_path = temp_dir_path / "metrics.json"
            shutil.copy2(metrics_path, temp_metrics_path)

            try:
                with open(metrics_path, "r", encoding="utf-8") as file:
                    loaded_metrics = json.load(file)

                if isinstance(loaded_metrics, dict):
                    raw_metrics = loaded_metrics

            except Exception as error:
                print(f"Could not read metrics_path for registry metrics: {error}")

        elif raw_metrics:
            temp_metrics_path = temp_dir_path / "metrics.json"
            with open(temp_metrics_path, "w", encoding="utf-8") as file:
                json.dump(raw_metrics, file, indent=4)

        registry_metrics = _flatten_metrics_for_hopsworks(raw_metrics)

        print("Trying to save model bundle to Hopsworks Model Registry...")
        print(f"Model registry name: {MODEL_NAME}")
        print(f"Model registry version: {MODEL_VERSION}")
        print(f"Model registry metrics: {registry_metrics}")

        create_model_kwargs = {
            "name": MODEL_NAME,
            "version": MODEL_VERSION,
            "metrics": registry_metrics,
            "description": description,
        }

        if metrics_path is not None and Path(metrics_path).exists():
            create_model_kwargs["metrics_path"] = str(metrics_path)

        try:
            model = model_registry.sklearn.create_model(**create_model_kwargs)

        except TypeError as error:
            print(f"Model Registry create_model did not accept all arguments: {error}")
            create_model_kwargs.pop("metrics_path", None)

            try:
                model = model_registry.sklearn.create_model(**create_model_kwargs)

            except TypeError:
                create_model_kwargs.pop("version", None)
                model = model_registry.sklearn.create_model(**create_model_kwargs)

        model.save(str(temp_dir_path))

        print("Model Registry status: saved successfully.")
        return True

    except Exception as error:
        print(f"Could not save to Hopsworks Model Registry: {error}")
        print("Model Registry status: skipped or failed safely. Local model backup is available.")
        return False

    finally:
        if temp_dir_path is not None and temp_dir_path.exists():
            shutil.rmtree(temp_dir_path, ignore_errors=True)


def load_local_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Local model not found at: {model_path}")

    return joblib.load(model_path)
