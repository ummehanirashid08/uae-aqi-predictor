import json
import os
import shutil
import tempfile
from pathlib import Path

import hopsworks
from dotenv import load_dotenv


load_dotenv(override=True)

PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME", "").strip()
API_KEY = os.getenv("HOPSWORKS_API_KEY", "").strip()

MODEL_PATH = Path("models/aqi_model.joblib")
METRICS_PATH = Path("models/metrics.json")

REMOTE_DIR = "Resources/aqi_model_artifacts"
REMOTE_MODEL_PATH = f"{REMOTE_DIR}/aqi_model.joblib"
REMOTE_METRICS_PATH = f"{REMOTE_DIR}/metrics.json"
REMOTE_README_PATH = f"{REMOTE_DIR}/README.md"


def is_missing_dataset_create_scope(error: Exception) -> bool:
    error_text = str(error)
    return "DATASET_CREATE" in error_text or "No valid scope found" in error_text


def print_dataset_create_scope_message() -> None:
    print(
        "Your Hopsworks API key is missing DATASET_CREATE scope. "
        "Create a new API key with DATASET_CREATE and update .env."
    )


def ensure_remote_folder(dataset_api) -> bool:
    # Hopsworks folder creation in the project filesystem requires an API key with
    # DATASET_CREATE scope. Without it, uploading to a new Resources subfolder fails.
    # This script intentionally uploads under Resources/aqi_model_artifacts and does
    # not attempt to create or use reserved datasets such as Models.
    print(f"Checking remote folder: {REMOTE_DIR}")

    try:
        if dataset_api.exists(REMOTE_DIR):
            print(f"Remote folder already exists: {REMOTE_DIR}")
            return True

    except Exception as error:
        print(f"Could not check whether remote folder exists: {error}")

    print(f"Creating remote folder if needed: {REMOTE_DIR}")

    try:
        dataset_api.mkdir(REMOTE_DIR)
        print(f"Remote folder ready: {REMOTE_DIR}")
        return True

    except Exception as error:
        if is_missing_dataset_create_scope(error):
            print_dataset_create_scope_message()
            return False

        print(f"Could not create remote folder {REMOTE_DIR}: {error}")
        return False


def main():
    if not PROJECT_NAME:
        print("HOPSWORKS_PROJECT_NAME is missing in .env")
        return

    if not API_KEY:
        print("HOPSWORKS_API_KEY is missing in .env")
        return

    if not MODEL_PATH.exists():
        print(f"Local model file not found: {MODEL_PATH}")
        return

    print("Connecting to Hopsworks...")
    try:
        project = hopsworks.login(project=PROJECT_NAME, api_key_value=API_KEY)
    except Exception as error:
        print(f"Could not connect to Hopsworks: {error}")
        return

    print(f"Connected to Hopsworks project: {PROJECT_NAME}")

    dataset_api = project.get_dataset_api()

    if not ensure_remote_folder(dataset_api):
        return

    temp_dir = Path(tempfile.mkdtemp(prefix="aqi_model_upload_"))

    try:
        temp_model = temp_dir / "aqi_model.joblib"
        temp_metrics = temp_dir / "metrics.json"
        temp_readme = temp_dir / "README.md"

        shutil.copy2(MODEL_PATH, temp_model)

        if METRICS_PATH.exists():
            shutil.copy2(METRICS_PATH, temp_metrics)
        else:
            temp_metrics.write_text(json.dumps({"status": "metrics file not found"}, indent=2), encoding="utf-8")

        temp_readme.write_text(
            "# UAE AQI Forecast Model Artifact\n\n"
            "This folder stores the trained AQI model artifact for the Weather AQI Predictor project.\n\n"
            "- Model file: `aqi_model.joblib`\n"
            "- Metrics file: `metrics.json`\n"
            "- Feature Store: Hopsworks Feature Store\n"
            "- Feature Group: `aqi_features_clean`, version `2`\n\n"
            "Note: The Hopsworks Model Registry system dataset was not available in this project, "
            "so the model artifact is stored in the Hopsworks Filesystem under Resources.\n",
            encoding="utf-8",
        )

        print("Uploading model artifact...")
        try:
            dataset_api.upload(str(temp_model), REMOTE_DIR, overwrite=True)
        except Exception as error:
            if is_missing_dataset_create_scope(error):
                print_dataset_create_scope_message()
            else:
                print(f"Could not upload model artifact: {error}")
            return

        print("Uploading metrics...")
        try:
            dataset_api.upload(str(temp_metrics), REMOTE_DIR, overwrite=True)
        except Exception as error:
            if is_missing_dataset_create_scope(error):
                print_dataset_create_scope_message()
            else:
                print(f"Could not upload metrics: {error}")
            return

        print("Uploading README...")
        try:
            dataset_api.upload(str(temp_readme), REMOTE_DIR, overwrite=True)
        except Exception as error:
            if is_missing_dataset_create_scope(error):
                print_dataset_create_scope_message()
            else:
                print(f"Could not upload README: {error}")
            return

        print("\nUpload successful.")
        print(f"Remote folder: {REMOTE_DIR}")
        print(f"Remote model: {REMOTE_MODEL_PATH}")
        print(f"Remote metrics: {REMOTE_METRICS_PATH}")
        print(f"Remote README: {REMOTE_README_PATH}")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
