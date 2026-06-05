import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    FEATURE_STORE_PATH,
    MODEL_PATH,
    METRICS_PATH,
)

from hopsworks_store import (
    load_features_from_hopsworks,
    save_model_to_hopsworks_registry,
)


class CurrentAQIBaseline(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame) and "us_aqi" in X.columns:
            return X["us_aqi"].values

        return np.asarray(X)[:, FEATURE_COLUMNS.index("us_aqi")]


def load_training_data():
    cloud_df, cloud_source = load_features_from_hopsworks()

    if cloud_df is not None and not cloud_df.empty:
        print(f"Training data loaded from: {cloud_source}")
        print(f"Cloud rows: {len(cloud_df)}")
        return cloud_df, cloud_source

    print(f"Could not load from Hopsworks: {cloud_source}")
    print("Falling back to local Parquet feature store...")

    if not FEATURE_STORE_PATH.exists():
        raise FileNotFoundError(
            f"Local feature store not found at {FEATURE_STORE_PATH}. "
            "Run feature_pipeline.py first."
        )

    local_df = pd.read_parquet(FEATURE_STORE_PATH)
    print(f"Training data loaded from local Parquet: {FEATURE_STORE_PATH}")
    print(f"Local rows: {len(local_df)}")

    return local_df, "Local Parquet"


def clean_training_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = df.copy()

    if "time" not in clean_df.columns:
        raise ValueError("Feature store must contain a 'time' column.")

    clean_df["time"] = pd.to_datetime(clean_df["time"], errors="coerce")
    clean_df = clean_df.dropna(subset=["time"])

    missing_features = [col for col in FEATURE_COLUMNS if col not in clean_df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    missing_targets = [col for col in TARGET_COLUMNS.values() if col not in clean_df.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")

    for column in FEATURE_COLUMNS + list(TARGET_COLUMNS.values()):
        clean_df[column] = pd.to_numeric(clean_df[column], errors="coerce")

    clean_df = clean_df.replace([np.inf, -np.inf], np.nan)
    clean_df = clean_df.sort_values("time").reset_index(drop=True)

    return clean_df


def get_model_candidates():
    return {
        "baseline_current_aqi": CurrentAQIBaseline(),

        "ridge": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),

        "lasso": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", Lasso(alpha=0.001, max_iter=10000)),
            ]
        ),

        "elastic_net": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000)),
            ]
        ),

        "random_forest": RandomForestRegressor(
            n_estimators=250,
            max_depth=18,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),

        "extra_trees": ExtraTreesRegressor(
            n_estimators=250,
            max_depth=22,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        ),

        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
        ),

        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=350,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=42,
        ),
    }


def time_based_split(target_df: pd.DataFrame, test_size: float = 0.2):
    target_df = target_df.sort_values("time").reset_index(drop=True)

    split_index = int(len(target_df) * (1 - test_size))

    if split_index <= 0 or split_index >= len(target_df):
        raise ValueError("Not enough rows for train/test split.")

    train_df = target_df.iloc[:split_index].copy()
    test_df = target_df.iloc[split_index:].copy()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["target"]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["target"]

    return X_train, X_test, y_train, y_test, train_df, test_df


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    mae = float(mean_absolute_error(y_test, predictions))
    r2 = float(r2_score(y_test, predictions))

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def train_models_for_target(clean_df: pd.DataFrame, target_key: str, target_column: str):
    print("\n" + "=" * 80)
    print(f"Training target: {target_key}")
    print(f"Target column: {target_column}")
    print("=" * 80)

    target_df = clean_df.dropna(subset=FEATURE_COLUMNS + [target_column]).copy()
    target_df = target_df.rename(columns={target_column: "target"})

    if len(target_df) < 200:
        raise ValueError(
            f"Not enough training rows for {target_key}. "
            f"Available rows after cleaning: {len(target_df)}"
        )

    print(f"Rows available for {target_key}: {len(target_df)}")

    X_train, X_test, y_train, y_test, train_df, test_df = time_based_split(target_df)

    print(f"Train rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")

    model_candidates = get_model_candidates()
    all_results = {}
    trained_models = {}

    for model_name, model in model_candidates.items():
        print(f"Training model: {model_name}")

        try:
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)

            all_results[model_name] = metrics
            trained_models[model_name] = model

            print(
                f"{model_name} | "
                f"RMSE: {metrics['rmse']:.4f} | "
                f"MAE: {metrics['mae']:.4f} | "
                f"R²: {metrics['r2']:.4f}"
            )

        except Exception as error:
            print(f"Model failed: {model_name}. Error: {error}")
            all_results[model_name] = {
                "error": str(error)
            }

    valid_results = {
        model_name: result
        for model_name, result in all_results.items()
        if isinstance(result, dict) and "rmse" in result
    }

    if not valid_results:
        raise RuntimeError(f"No valid models trained for target {target_key}")

    best_model_name = min(valid_results, key=lambda name: valid_results[name]["rmse"])
    best_model = trained_models[best_model_name]
    best_metrics = valid_results[best_model_name]

    print(f"Best model for {target_key}: {best_model_name}")
    print(f"Best metrics: {best_metrics}")

    return {
        "target_key": target_key,
        "target_column": target_column,
        "best_model": best_model_name,
        "best_metrics": best_metrics,
        "all_model_results": all_results,
        "model": best_model,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "total_rows": int(len(target_df)),
        "test_start_time": str(test_df["time"].min()),
        "test_end_time": str(test_df["time"].max()),
    }


def build_registry_metrics(metrics_payload: dict):
    registry_metrics = {}

    targets = metrics_payload.get("targets", {})

    for target_key, target_data in targets.items():
        best_metrics = target_data.get("best_metrics", {})

        for metric_name, metric_value in best_metrics.items():
            if isinstance(metric_value, (int, float)):
                safe_key = f"{target_key}_{metric_name}"
                registry_metrics[safe_key] = float(metric_value)

    return registry_metrics


def save_training_outputs(model_bundle: dict, metrics_payload: dict):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_bundle, MODEL_PATH)

    with open(METRICS_PATH, "w") as file:
        json.dump(metrics_payload, file, indent=4)

    print("\nTraining outputs saved locally.")
    print(f"Model path: {MODEL_PATH}")
    print(f"Metrics path: {METRICS_PATH}")


def main():
    print("Starting AQI training pipeline...")

    raw_df, data_source = load_training_data()
    clean_df = clean_training_dataframe(raw_df)

    print(f"Clean dataframe rows: {len(clean_df)}")
    print(f"Training data source: {data_source}")
    print(f"Feature columns: {len(FEATURE_COLUMNS)}")

    model_bundle = {
        "trained_at": datetime.utcnow().isoformat(),
        "data_source": data_source,
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "models": {},
    }

    metrics_payload = {
        "trained_at": datetime.utcnow().isoformat(),
        "data_source": data_source,
        "feature_columns_count": len(FEATURE_COLUMNS),
        "rows_total": int(len(clean_df)),
        "targets": {},
    }

    for target_key, target_column in TARGET_COLUMNS.items():
        target_result = train_models_for_target(clean_df, target_key, target_column)

        model_bundle["models"][target_key] = {
            "model_name": target_result["best_model"],
            "model": target_result["model"],
            "target_column": target_column,
        }

        metrics_payload["targets"][target_key] = {
            "target_column": target_column,
            "best_model": target_result["best_model"],
            "best_metrics": target_result["best_metrics"],
            "all_model_results": target_result["all_model_results"],
            "train_rows": target_result["train_rows"],
            "test_rows": target_result["test_rows"],
            "total_rows": target_result["total_rows"],
            "test_start_time": target_result["test_start_time"],
            "test_end_time": target_result["test_end_time"],
        }

    save_training_outputs(model_bundle, metrics_payload)

    registry_metrics = build_registry_metrics(metrics_payload)

    print("\nTrying to save model bundle to Hopsworks Model Registry...")
    registry_saved = save_model_to_hopsworks_registry(
        model_path=MODEL_PATH,
        metrics=registry_metrics,
        description="Multi-target AQI forecasting model bundle for UAE cities. Predicts Day 1, Day 2, Day 3, and 3-day average AQI.",
    )

    if registry_saved:
        print("Model Registry status: saved successfully.")
    else:
        print("Model Registry status: skipped or failed safely. Local model backup is available.")

    print("Training pipeline completed.")


if __name__ == "__main__":
    main()