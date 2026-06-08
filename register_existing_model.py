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
MODEL_NAME = os.getenv("MODEL_NAME", "uae_aqi_forecast_model").strip()
MODEL_VERSION = int(os.getenv("MODEL_VERSION", "1"))

MODEL_PATH = Path("models/aqi_model.joblib")
METRICS_PATH = Path("models/metrics.json")


def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        print(f"Metrics file not found: {METRICS_PATH}")
        return {}

    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            metrics = json.load(file)

        flat_metrics = {}

        if isinstance(metrics, dict):
            if "targets" in metrics:
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

    except Exception as error:
        print(f"Could not read metrics: {error}")
        return {}


def main():
    if not PROJECT_NAME:
        raise RuntimeError("HOPSWORKS_PROJECT_NAME is missing in .env")

    if not API_KEY:
        raise RuntimeError("HOPSWORKS_API_KEY is missing in .env")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    print("Connecting to Hopsworks...")
    project = hopsworks.login(
        project=PROJECT_NAME,
        api_key_value=API_KEY,
    )

    print("Connected to Hopsworks project:", PROJECT_NAME)

    print("Opening Model Registry...")
    model_registry = project.get_model_registry()

    metrics = load_metrics()

    temp_dir = Path(tempfile.mkdtemp(prefix="aqi_existing_model_"))

    try:
        model_file = temp_dir / MODEL_PATH.name
        shutil.copy2(MODEL_PATH, model_file)

        if METRICS_PATH.exists():
            shutil.copy2(METRICS_PATH, temp_dir / "metrics.json")

        readme_path = temp_dir / "README.md"
        readme_path.write_text(
            "# UAE AQI Forecast Model\n\n"
            "This model bundle predicts AQI for Day 1, Day 2, Day 3, and 3-Day Average.\n\n"
            "Registered from local trained artifact without re-running full training.\n",
            encoding="utf-8",
        )

        print("Creating model registry entry...")
        print("Model name:", MODEL_NAME)
        print("Model version:", MODEL_VERSION)
        print("Model folder:", temp_dir)
        print("Metrics:", metrics)

        create_kwargs = {
            "name": MODEL_NAME,
            "version": MODEL_VERSION,
            "metrics": metrics,
            "description": "Multi-horizon UAE AQI forecasting model bundle registered from local artifact.",
        }

        try:
            model = model_registry.sklearn.create_model(**create_kwargs)

        except TypeError as error:
            print("Version argument not supported by this Hopsworks client, retrying without version.")
            print("Original error:", error)
            create_kwargs.pop("version", None)
            model = model_registry.sklearn.create_model(**create_kwargs)

        print("Uploading model artifact to Hopsworks Model Registry...")
        model.save(str(temp_dir))

        print("Model Registry upload successful.")
        print("Now refresh Hopsworks Model Registry page.")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()