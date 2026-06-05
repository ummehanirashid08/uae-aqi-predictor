import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import joblib
import pandas as pd
import hopsworks

from config import (
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT_NAME,
    HOPSWORKS_FEATURE_GROUP_NAME,
    HOPSWORKS_FEATURE_GROUP_VERSION,
    HOPSWORKS_MODEL_NAME,
)


def prepare_windows_temp_dir() -> None:
    """
    Hopsworks client uses temporary certificate folders.
    On Windows, /tmp may not exist, so this forces temp paths to C:\\tmp.
    """
    if os.name == "nt":
        tmp_dir = Path("C:/tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        os.environ["TMP"] = str(tmp_dir)
        os.environ["TEMP"] = str(tmp_dir)
        tempfile.tempdir = str(tmp_dir)


def is_hopsworks_enabled() -> bool:
    return os.getenv("USE_HOPSWORKS", "false").lower() == "true"


def validate_hopsworks_env() -> None:
    if not HOPSWORKS_API_KEY:
        raise ValueError("HOPSWORKS_API_KEY is missing in .env")

    if not HOPSWORKS_PROJECT_NAME:
        raise ValueError("HOPSWORKS_PROJECT_NAME is missing in .env")


def connect_to_hopsworks():
    prepare_windows_temp_dir()
    validate_hopsworks_env()

    project = hopsworks.login(
        project=HOPSWORKS_PROJECT_NAME,
        api_key_value=HOPSWORKS_API_KEY,
    )

    return project


def get_feature_store():
    project = connect_to_hopsworks()
    return project.get_feature_store()


def normalize_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = df.copy()

    if "time" not in clean_df.columns:
        raise ValueError("Feature dataframe must contain a 'time' column.")

    if "latitude" not in clean_df.columns or "longitude" not in clean_df.columns:
        raise ValueError("Feature dataframe must contain latitude and longitude columns.")

    clean_df["time"] = pd.to_datetime(clean_df["time"])

    clean_df["location_key"] = (
        clean_df["latitude"].round(4).astype(str)
        + "_"
        + clean_df["longitude"].round(4).astype(str)
    )

    if "city" in clean_df.columns:
        clean_df["city"] = clean_df["city"].astype(str)

    for column in clean_df.columns:
        if clean_df[column].dtype == "object" and column not in ["city", "location_key"]:
            try:
                clean_df[column] = pd.to_numeric(clean_df[column])
            except Exception:
                clean_df[column] = clean_df[column].astype(str)

    clean_df = clean_df.replace([float("inf"), float("-inf")], pd.NA)

    return clean_df


def get_existing_feature_group(fs):
    try:
        feature_group = fs.get_feature_group(
            name=HOPSWORKS_FEATURE_GROUP_NAME,
            version=HOPSWORKS_FEATURE_GROUP_VERSION,
        )

        if feature_group is not None:
            print(
                f"Using existing Hopsworks feature group: "
                f"{HOPSWORKS_FEATURE_GROUP_NAME}, version {HOPSWORKS_FEATURE_GROUP_VERSION}"
            )
            return feature_group

        return None

    except Exception as error:
        print("Existing feature group not found or not accessible:", error)
        return None


def create_feature_group(fs):
    print(
        f"Creating Hopsworks feature group: "
        f"{HOPSWORKS_FEATURE_GROUP_NAME}, version {HOPSWORKS_FEATURE_GROUP_VERSION}"
    )

    feature_group = fs.get_or_create_feature_group(
        name=HOPSWORKS_FEATURE_GROUP_NAME,
        version=HOPSWORKS_FEATURE_GROUP_VERSION,
        description=(
            "Processed AQI, pollutant, weather, lag, rolling, "
            "and future weather features for UAE AQI forecasting."
        ),
        primary_key=["location_key", "time"],
        event_time="time",
        online_enabled=False,
    )

    if feature_group is None:
        raise RuntimeError("Hopsworks returned None while creating feature group.")

    return feature_group


def get_or_create_feature_group(fs):
    feature_group = get_existing_feature_group(fs)

    if feature_group is not None:
        return feature_group

    feature_group = create_feature_group(fs)

    return feature_group


def insert_feature_dataframe(feature_group, feature_df: pd.DataFrame) -> None:
    if feature_group is None:
        raise RuntimeError("Feature group is None. Cannot insert features.")

    print("Uploading features to Hopsworks Feature Store...")

    try:
        feature_group.insert(
            feature_df,
            write_options={
                "wait_for_job": True,
            },
        )
        return

    except TypeError:
        feature_group.insert(
            feature_df,
            wait=True,
        )
        return


def save_features_to_hopsworks(df: pd.DataFrame) -> bool:
    if not is_hopsworks_enabled():
        print("USE_HOPSWORKS=false. Skipping Hopsworks feature save.")
        return False

    try:
        feature_df = normalize_feature_dataframe(df)

        print(f"Prepared dataframe for Hopsworks upload: {feature_df.shape[0]} rows, {feature_df.shape[1]} columns")

        fs = get_feature_store()
        feature_group = get_or_create_feature_group(fs)

        insert_feature_dataframe(feature_group, feature_df)

        print("Hopsworks Feature Store upload completed.")
        return True

    except Exception as error:
        print("Hopsworks feature save failed:", error)
        return False


def load_features_from_hopsworks() -> Tuple[Optional[pd.DataFrame], str]:
    if not is_hopsworks_enabled():
        return None, "Hopsworks disabled"

    try:
        fs = get_feature_store()

        feature_group = fs.get_feature_group(
            name=HOPSWORKS_FEATURE_GROUP_NAME,
            version=HOPSWORKS_FEATURE_GROUP_VERSION,
        )

        if feature_group is None:
            return None, "Hopsworks feature group not found"

        print("Reading features from Hopsworks Feature Store...")

        query = feature_group.select_all()
        df = query.read()

        if df is None or df.empty:
            return None, "Hopsworks Feature Store returned empty dataframe"

        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)

        print(f"Loaded {len(df)} rows from Hopsworks Feature Store.")
        return df, "Hopsworks Feature Store"

    except Exception as error:
        print("Hopsworks feature load failed:", error)
        return None, f"Hopsworks load failed: {error}"


def save_model_to_hopsworks_registry(
    model_path: Path,
    metrics: dict,
    description: str = "AQI forecasting model bundle",
) -> bool:
    """
    Saves model artifact to Hopsworks Model Registry.
    If Model Registry is not initialized, this fails safely and local model remains available.
    """
    if not is_hopsworks_enabled():
        print("USE_HOPSWORKS=false. Skipping Hopsworks model registry save.")
        return False

    try:
        project = connect_to_hopsworks()
        model_registry = project.get_model_registry()

        model = model_registry.sklearn.create_model(
            name=HOPSWORKS_MODEL_NAME,
            metrics=metrics,
            description=description,
        )

        model.save(str(model_path.parent))

        print("Model saved to Hopsworks Model Registry.")
        return True

    except Exception as error:
        print("Hopsworks Model Registry save failed:", error)
        print("Continuing with local model backup only.")
        return False


def load_local_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Local model not found at: {model_path}")

    return joblib.load(model_path)