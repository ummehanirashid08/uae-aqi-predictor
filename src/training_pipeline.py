import json
import os
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import FEATURE_STORE_PATH, MODEL_PATH, METRICS_PATH
from hopsworks_store import load_features_from_hopsworks, save_model_to_hopsworks_registry


os.environ["LOKY_MAX_CPU_COUNT"] = "4"


REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


TARGETS = {
    "day_1": {
        "target_column": "target_aqi_day1_avg",
        "label": "Day 1 AQI Average",
    },
    "day_2": {
        "target_column": "target_aqi_day2_avg",
        "label": "Day 2 AQI Average",
    },
    "day_3": {
        "target_column": "target_aqi_day3_avg",
        "label": "Day 3 AQI Average",
    },
    "next_72h_avg": {
        "target_column": "target_aqi_next_72h_avg",
        "label": "Next 72 Hours AQI Average",
    },
}


NON_FEATURE_COLUMNS = {
    "time",
    "city",
    "location_key",
    "location_lookup_key",
    "created_at",
    "updated_at",
    "date",
    "timestamp",
}


TARGET_COLUMNS = {
    "target_aqi_day1_avg",
    "target_aqi_day2_avg",
    "target_aqi_day3_avg",
    "target_aqi_next_72h_avg",
}


def safe_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def safe_r2(y_true, y_pred):
    try:
        return float(r2_score(y_true, y_pred))
    except Exception:
        return None


def load_training_data():
    cloud_df, cloud_source = load_features_from_hopsworks()

    if cloud_df is not None and not cloud_df.empty:
        print(f"Training data loaded from Hopsworks Feature Store: {len(cloud_df)} rows")
        cloud_df["time"] = pd.to_datetime(cloud_df["time"], errors="coerce")
        cloud_df = cloud_df.dropna(subset=["time"])
        cloud_df = cloud_df.sort_values("time").reset_index(drop=True)
        return cloud_df, "Hopsworks Feature Store"

    print(f"Could not load from Hopsworks Feature Store: {cloud_source}")
    print("Falling back to local Parquet feature store...")

    if not FEATURE_STORE_PATH.exists():
        raise FileNotFoundError(
            f"Local feature store not found at {FEATURE_STORE_PATH}. "
            "Run feature_pipeline.py first."
        )

    local_df = pd.read_parquet(FEATURE_STORE_PATH)

    if local_df.empty:
        raise RuntimeError("Local feature store is empty. Run feature_pipeline.py again.")

    local_df["time"] = pd.to_datetime(local_df["time"], errors="coerce")
    local_df = local_df.dropna(subset=["time"])
    local_df = local_df.sort_values("time").reset_index(drop=True)

    print(f"Training data loaded from local Parquet: {len(local_df)} rows")

    return local_df, "Local Parquet"


def normalize_columns(df):
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


def clean_training_dataframe(df):
    df = normalize_columns(df)

    if "time" not in df.columns:
        raise ValueError("Training dataframe must contain a time column.")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])

    if "city" in df.columns:
        df["city"] = df["city"].astype(str).str.strip()
        df["city"] = df["city"].replace(["", "None", "none", "nan", "NaN", "NULL", "null"], "unknown")
    else:
        df["city"] = "unknown"

    df = df.drop_duplicates(subset=["time", "city"], keep="last")
    df = df.sort_values(["city", "time"]).reset_index(drop=True)

    df = df.replace([np.inf, -np.inf], np.nan)

    for column in df.columns:
        if column not in {"time", "city", "location_key", "location_lookup_key"}:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def get_candidate_feature_columns(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    feature_columns = []

    for column in numeric_columns:
        if column in NON_FEATURE_COLUMNS:
            continue

        if column in TARGET_COLUMNS:
            continue

        if column.startswith("target_"):
            continue

        if df[column].notna().sum() == 0:
            continue

        feature_columns.append(column)

    return feature_columns


def add_city_one_hot_features(df):
    df = df.copy()

    if "city" not in df.columns:
        return df, []

    city_dummies = pd.get_dummies(df["city"], prefix="city", dtype=int)

    df = pd.concat([df, city_dummies], axis=1)

    return df, city_dummies.columns.tolist()


def remove_high_missing_features(df, feature_columns, missing_threshold=0.95):
    selected_features = []

    for column in feature_columns:
        missing_ratio = df[column].isna().mean()

        if missing_ratio <= missing_threshold:
            selected_features.append(column)

    removed_count = len(feature_columns) - len(selected_features)

    if removed_count > 0:
        print(f"Removed {removed_count} features with more than {missing_threshold * 100:.0f}% missing values.")

    return selected_features


def make_models():
    models = {
        "ridge": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "elastic_net": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=0.01, l1_ratio=0.2, random_state=42, max_iter=5000)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=220,
                        max_depth=18,
                        min_samples_split=4,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=260,
                        max_depth=None,
                        min_samples_split=3,
                        min_samples_leaf=1,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=260,
                        learning_rate=0.035,
                        max_depth=4,
                        min_samples_split=4,
                        min_samples_leaf=2,
                        subsample=0.9,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=350,
                        learning_rate=0.035,
                        max_leaf_nodes=31,
                        min_samples_leaf=12,
                        l2_regularization=0.05,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    try:
        from xgboost import XGBRegressor

        models["xgboost"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=320,
                        learning_rate=0.035,
                        max_depth=4,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="reg:squarederror",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    except Exception:
        print("XGBoost not installed. Skipping xgboost model.")

    return models


def temporal_train_test_split(df, target_column, feature_columns, test_ratio=0.2):
    working_df = df.copy()

    working_df = working_df.dropna(subset=[target_column])

    if working_df.empty:
        return None, None, None, None, working_df

    working_df = working_df.sort_values("time").reset_index(drop=True)

    X = working_df[feature_columns].copy()
    y = working_df[target_column].copy()

    for column in X.columns:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    y = pd.to_numeric(y, errors="coerce")

    valid_target_mask = y.notna()

    X = X.loc[valid_target_mask].reset_index(drop=True)
    y = y.loc[valid_target_mask].reset_index(drop=True)
    working_df = working_df.loc[valid_target_mask].reset_index(drop=True)

    if len(working_df) < 100:
        return None, None, None, None, working_df

    split_index = int(len(working_df) * (1 - test_ratio))

    if split_index < 50:
        split_index = max(1, int(len(working_df) * 0.7))

    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()
    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()

    if len(X_test) < 10:
        return None, None, None, None, working_df

    return X_train, X_test, y_train, y_test, working_df


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    predictions = np.clip(predictions, 0, 500)

    rmse = safe_rmse(y_test, predictions)
    mae = safe_mae(y_test, predictions)
    r2 = safe_r2(y_test, predictions)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "predictions": predictions,
    }


def train_target_model(df, target_key, target_column, feature_columns):
    print("")
    print(f"Training target: {target_key} ({target_column})")

    if target_column not in df.columns:
        print(f"Skipping {target_key}: target column not found.")
        return None

    X_train, X_test, y_train, y_test, clean_df = temporal_train_test_split(
        df=df,
        target_column=target_column,
        feature_columns=feature_columns,
    )

    if X_train is None:
        print(f"Skipping {target_key}: not enough clean rows ({len(clean_df)}).")
        return None

    print(f"Clean rows for {target_key}: {len(clean_df)}")
    print(f"Train rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Features used: {len(feature_columns)}")

    models = make_models()

    all_model_results = {}
    trained_models = {}

    baseline_predictions = np.repeat(float(y_train.iloc[-1]), len(y_test))

    all_model_results["baseline_current_aqi"] = {
        "rmse": safe_rmse(y_test, baseline_predictions),
        "mae": safe_mae(y_test, baseline_predictions),
        "r2": safe_r2(y_test, baseline_predictions),
    }

    for model_name, model in models.items():
        try:
            result = evaluate_model(
                model=model,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
            )

            all_model_results[model_name] = {
                "rmse": result["rmse"],
                "mae": result["mae"],
                "r2": result["r2"],
            }

            trained_models[model_name] = model

            print(
                f"{model_name}: "
                f"RMSE={result['rmse']:.4f}, "
                f"MAE={result['mae']:.4f}, "
                f"R2={result['r2']:.4f}"
            )

        except Exception as error:
            print(f"{model_name} failed: {error}")

    model_results_without_baseline = {
        name: values
        for name, values in all_model_results.items()
        if name != "baseline_current_aqi" and values.get("r2") is not None
    }

    if not model_results_without_baseline:
        print(f"No valid trained models for {target_key}.")
        return None

    best_model_name = sorted(
        model_results_without_baseline.items(),
        key=lambda item: (
            item[1].get("r2") if item[1].get("r2") is not None else -999,
            -item[1].get("rmse", 999),
        ),
        reverse=True,
    )[0][0]

    best_model = trained_models[best_model_name]
    best_metrics = all_model_results[best_model_name]

    print(f"Best model for {target_key}: {best_model_name}")
    print(f"Best metrics: {best_metrics}")

    return {
        "target_key": target_key,
        "target_column": target_column,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "best_metrics": best_metrics,
        "all_model_results": all_model_results,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "clean_rows": int(len(clean_df)),
        "feature_columns": feature_columns,
    }


def save_model_bundle(model_bundle):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, MODEL_PATH)
    print(f"Saved model bundle to: {MODEL_PATH}")


def save_metrics(metrics):
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"Saved metrics to: {METRICS_PATH}")


def save_training_summary(metrics):
    rows = []

    for target_key, target_data in metrics.get("targets", {}).items():
        all_models = target_data.get("all_models", {})

        for model_name, model_metrics in all_models.items():
            rows.append(
                {
                    "target_key": target_key,
                    "target_column": target_data.get("target_column"),
                    "model": model_name,
                    "rmse": model_metrics.get("rmse"),
                    "mae": model_metrics.get("mae"),
                    "r2": model_metrics.get("r2"),
                    "best_model": target_data.get("best_model"),
                    "train_rows": target_data.get("train_rows"),
                    "test_rows": target_data.get("test_rows"),
                    "clean_rows": target_data.get("clean_rows"),
                }
            )

    summary_df = pd.DataFrame(rows)

    summary_path = REPORTS_DIR / "model_training_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved training summary to: {summary_path}")


def main():
    print("Starting AQI training pipeline...")

    raw_df, data_source = load_training_data()
    df = clean_training_dataframe(raw_df)

    df, city_feature_columns = add_city_one_hot_features(df)

    base_feature_columns = get_candidate_feature_columns(df)
    feature_columns = sorted(list(set(base_feature_columns + city_feature_columns)))

    feature_columns = remove_high_missing_features(df, feature_columns)

    print(f"Training data source: {data_source}")
    print(f"Training dataframe shape: {df.shape}")
    print(f"Initial feature count: {len(feature_columns)}")

    if not feature_columns:
        raise RuntimeError("No usable feature columns found.")

    trained_target_models = {}
    metrics_targets = {}

    for target_key, target_info in TARGETS.items():
        target_column = target_info["target_column"]

        result = train_target_model(
            df=df,
            target_key=target_key,
            target_column=target_column,
            feature_columns=feature_columns,
        )

        if result is None:
            continue

        trained_target_models[target_key] = {
            "model": result["best_model"],
            "model_name": result["best_model_name"],
            "target_column": target_column,
            "feature_columns": result["feature_columns"],
        }

        metrics_targets[target_key] = {
            "target_key": target_key,
            "target_column": target_column,
            "label": target_info["label"],
            "best_model": result["best_model_name"],
            "best_metrics": result["best_metrics"],
            "all_models": result["all_model_results"],
            "all_model_results": result["all_model_results"],
            "train_rows": result["train_rows"],
            "test_rows": result["test_rows"],
            "clean_rows": result["clean_rows"],
            "feature_count": len(result["feature_columns"]),
        }

    if not trained_target_models:
        raise RuntimeError("No models were trained successfully.")

    model_bundle = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": data_source,
        "model_type": "multi_target_aqi_forecast",
        "aqi_standard": "AQI-US",
        "feature_columns": feature_columns,
        "targets": TARGETS,
        "models": trained_target_models,
    }

    metrics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": data_source,
        "model_path": str(MODEL_PATH),
        "metrics_path": str(METRICS_PATH),
        "feature_store_path": str(FEATURE_STORE_PATH),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "targets": metrics_targets,
    }

    save_model_bundle(model_bundle)
    save_metrics(metrics)
    save_training_summary(metrics)

    print("Trying to save model bundle to Hopsworks Model Registry...")

    registry_saved = save_model_to_hopsworks_registry(
        model_path=MODEL_PATH,
        metrics_path=METRICS_PATH,
        description=(
            "Multi-target AQI forecasting model bundle for UAE cities. "
            "Predicts Day 1, Day 2, Day 3, and 3-day average AQI using cleaned pollutant, "
            "weather, lag, rolling, and future weather features."
        ),
    )

    if registry_saved:
        print("Model Registry status: saved successfully.")
    else:
        print("Model Registry status: skipped or failed safely. Local model backup is available.")

    print("Training pipeline completed.")


if __name__ == "__main__":
    main()
