import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn.inspection import permutation_importance

try:
    import shap

    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False

from config import FEATURE_COLUMNS, FEATURE_STORE_PATH, MODEL_PATH, METRICS_PATH
from hopsworks_store import load_features_from_hopsworks


REPORTS_DIR = Path("reports")

EDA_HTML_PATH = REPORTS_DIR / "eda_report.html"
EDA_SUMMARY_JSON_PATH = REPORTS_DIR / "eda_summary.json"

EXPLAINABILITY_HTML_PATH = REPORTS_DIR / "explainability_report.html"
EXPLAINABILITY_CSV_PATH = REPORTS_DIR / "feature_importance.csv"

SHAP_HTML_PATH = REPORTS_DIR / "shap_report.html"
SHAP_CSV_PATH = REPORTS_DIR / "shap_feature_importance.csv"


CITY_MAP = {
    "24.1302,55.8023": "Al Ain",
    "24.4539,54.3773": "Abu Dhabi",
    "25.2048,55.2708": "Dubai",
    "25.3463,55.4209": "Sharjah",
    "25.4052,55.5136": "Ajman",
    "25.8007,55.9762": "Ras Al Khaimah",
}


TARGET_COLUMNS = {
    "day_1": "target_aqi_day1_avg",
    "day_2": "target_aqi_day2_avg",
    "day_3": "target_aqi_day3_avg",
    "next_72h_avg": "target_aqi_next_72h_avg",
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


def ensure_directories():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def make_json_safe(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}

    if isinstance(value, list):
        return [make_json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)

    return value


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
        df["city"] = df["city"].replace(
            ["None", "none", "nan", "NaN", "NULL", "null", ""],
            pd.NA,
        )
        df["city"] = df["city"].fillna(mapped_city)
    else:
        df["city"] = mapped_city

    df["city"] = df["city"].fillna(df["location_lookup_key"])
    df["city"] = df["city"].astype(str)
    df = df[df["city"].str.lower() != "none"].copy()

    return df


def load_feature_data() -> tuple[pd.DataFrame, str]:
    cloud_df, cloud_source = load_features_from_hopsworks()

    if cloud_df is not None and not cloud_df.empty:
        cloud_df["time"] = pd.to_datetime(cloud_df["time"], errors="coerce")
        cloud_df = cloud_df.dropna(subset=["time"])
        cloud_df = cloud_df.sort_values("time").reset_index(drop=True)
        cloud_df = create_city_column(cloud_df)
        return cloud_df, cloud_source

    if not FEATURE_STORE_PATH.exists():
        raise FileNotFoundError(
            f"Feature store not found at {FEATURE_STORE_PATH}. "
            "Run feature_pipeline.py first or enable Hopsworks."
        )

    local_df = pd.read_parquet(FEATURE_STORE_PATH)
    local_df["time"] = pd.to_datetime(local_df["time"], errors="coerce")
    local_df = local_df.dropna(subset=["time"])
    local_df = local_df.sort_values("time").reset_index(drop=True)
    local_df = create_city_column(local_df)

    return local_df, "Local Parquet Feature Store"


def load_model_bundle():
    if not MODEL_PATH.exists():
        print("Model file not found. Explainability and SHAP reports will be skipped.")
        return None

    bundle = joblib.load(MODEL_PATH)

    if not isinstance(bundle, dict) or "models" not in bundle:
        print("Old model format found. Explainability and SHAP reports will be skipped.")
        return None

    return bundle


def load_metrics():
    if not METRICS_PATH.exists():
        return None

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def safe_round(value, decimals=2):
    try:
        if pd.isna(value):
            return None
    except Exception:
        return None

    return round(float(value), decimals)


def write_html_report(title: str, sections: list[str], output_path: Path):
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    body = "\n".join(sections)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{title}</title>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background: #f8fafc;
                font-family: Arial, sans-serif;
                color: #0f172a;
            }}

            .container {{
                max-width: 1180px;
                margin: 0 auto;
                padding: 34px 24px 60px;
            }}

            .hero {{
                background: linear-gradient(135deg, #fff8d8 0%, #facc15 100%);
                border: 1px solid #fde68a;
                border-radius: 28px;
                padding: 34px;
                box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
                margin-bottom: 24px;
            }}

            .hero h1 {{
                margin: 0 0 10px;
                font-size: 38px;
                letter-spacing: -1px;
            }}

            .hero p {{
                margin: 0;
                color: #475569;
                font-size: 16px;
                font-weight: 700;
            }}

            .card {{
                background: #fffdf2;
                border: 1px solid #f3e8a2;
                border-radius: 24px;
                padding: 24px;
                box-shadow: 0 14px 34px rgba(15, 23, 42, 0.06);
                margin-bottom: 24px;
            }}

            .grid {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 16px;
                margin-bottom: 24px;
            }}

            .metric {{
                background: white;
                border: 1px solid #e5e7eb;
                border-radius: 18px;
                padding: 18px;
            }}

            .metric-title {{
                color: #64748b;
                font-size: 13px;
                font-weight: 800;
                margin-bottom: 8px;
            }}

            .metric-value {{
                font-size: 28px;
                font-weight: 900;
                color: #0f172a;
            }}

            h2 {{
                font-size: 26px;
                margin: 0 0 16px;
                letter-spacing: -0.7px;
            }}

            .note {{
                background: #eff6ff;
                color: #1e40af;
                border: 1px solid #bfdbfe;
                border-radius: 18px;
                padding: 16px;
                font-weight: 700;
                line-height: 1.55;
                margin-bottom: 24px;
            }}

            .warning {{
                background: #fff7ed;
                color: #9a3412;
                border: 1px solid #fed7aa;
                border-radius: 18px;
                padding: 16px;
                font-weight: 700;
                line-height: 1.55;
                margin-bottom: 24px;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                background: white;
                border-radius: 16px;
                overflow: hidden;
            }}

            th, td {{
                text-align: left;
                border-bottom: 1px solid #e5e7eb;
                padding: 12px 14px;
                font-size: 14px;
            }}

            th {{
                background: #f1f5f9;
                font-weight: 900;
                color: #334155;
            }}

            pre {{
                white-space: pre-wrap;
                background: #0f172a;
                color: #e5e7eb;
                border-radius: 18px;
                padding: 18px;
                overflow-x: auto;
            }}

            @media (max-width: 900px) {{
                .grid {{
                    grid-template-columns: repeat(2, 1fr);
                }}
            }}

            @media (max-width: 600px) {{
                .grid {{
                    grid-template-columns: 1fr;
                }}

                .hero h1 {{
                    font-size: 30px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="hero">
                <h1>{title}</h1>
                <p>Generated automatically on {generated_at}</p>
            </div>
            {body}
        </div>
    </body>
    </html>
    """

    output_path.write_text(html, encoding="utf-8")


def plot_to_html(fig):
    return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")


def style_fig(fig, height=450):
    fig.update_layout(
        height=height,
        plot_bgcolor="#fffdf2",
        paper_bgcolor="#fffdf2",
        font=dict(color="#0f172a", family="Arial, sans-serif"),
        title_font=dict(size=22, color="#0f172a"),
        margin=dict(l=30, r=30, t=70, b=40),
    )

    return fig


def generate_eda_report(df: pd.DataFrame, data_source: str):
    print("Generating EDA report...")

    total_rows = len(df)
    total_cities = df["city"].nunique()
    min_time = df["time"].min()
    max_time = df["time"].max()
    avg_aqi = df["us_aqi"].mean()
    max_aqi = df["us_aqi"].max()

    city_summary = (
        df.groupby("city")
        .agg(
            rows=("us_aqi", "count"),
            avg_aqi=("us_aqi", "mean"),
            max_aqi=("us_aqi", "max"),
            min_aqi=("us_aqi", "min"),
            latest_time=("time", "max"),
        )
        .reset_index()
        .sort_values("avg_aqi", ascending=False)
    )

    city_summary["avg_aqi"] = city_summary["avg_aqi"].round(2)
    city_summary["max_aqi"] = city_summary["max_aqi"].round(2)
    city_summary["min_aqi"] = city_summary["min_aqi"].round(2)
    city_summary["latest_time"] = city_summary["latest_time"].astype(str)

    available_pollutants = [col for col in POLLUTANT_COLUMNS if col in df.columns]
    available_targets = [col for col in TARGET_COLUMNS.values() if col in df.columns]

    trend_df = (
        df.groupby([pd.Grouper(key="time", freq="D"), "city"])["us_aqi"]
        .mean()
        .reset_index()
    )

    trend_fig = px.line(
        trend_df,
        x="time",
        y="us_aqi",
        color="city",
        title="Daily Average AQI Trend by City",
        markers=False,
    )
    trend_fig = style_fig(trend_fig, height=520)

    city_bar_fig = px.bar(
        city_summary,
        x="city",
        y="avg_aqi",
        title="Average AQI by City",
        text="avg_aqi",
    )
    city_bar_fig.update_traces(textposition="outside", marker_color="#2563eb")
    city_bar_fig = style_fig(city_bar_fig, height=480)

    aqi_hist_fig = px.histogram(
        df,
        x="us_aqi",
        nbins=40,
        color="city",
        title="AQI Distribution by City",
    )
    aqi_hist_fig = style_fig(aqi_hist_fig, height=480)

    sections = []

    sections.append(
        f"""
        <div class="grid">
            <div class="metric">
                <div class="metric-title">Total Rows</div>
                <div class="metric-value">{total_rows:,}</div>
            </div>
            <div class="metric">
                <div class="metric-title">Cities</div>
                <div class="metric-value">{total_cities}</div>
            </div>
            <div class="metric">
                <div class="metric-title">Average AQI</div>
                <div class="metric-value">{safe_round(avg_aqi, 1)}</div>
            </div>
            <div class="metric">
                <div class="metric-title">Maximum AQI</div>
                <div class="metric-value">{safe_round(max_aqi, 1)}</div>
            </div>
        </div>

        <div class="note">
            <b>Data source:</b> {data_source}<br>
            <b>Time range:</b> {str(min_time)} to {str(max_time)}
        </div>
        """
    )

    sections.append(
        f"""
        <div class="card">
            <h2>City Summary</h2>
            {city_summary.to_html(index=False)}
        </div>
        """
    )

    sections.append(
        f"""
        <div class="card">
            <h2>AQI Trend</h2>
            {plot_to_html(trend_fig)}
        </div>
        """
    )

    sections.append(
        f"""
        <div class="card">
            <h2>Average AQI by City</h2>
            {plot_to_html(city_bar_fig)}
        </div>
        """
    )

    sections.append(
        f"""
        <div class="card">
            <h2>AQI Distribution</h2>
            {plot_to_html(aqi_hist_fig)}
        </div>
        """
    )

    if available_pollutants:
        pollutant_avg = (
            df.groupby("city")[available_pollutants]
            .mean()
            .reset_index()
            .round(2)
        )

        pollutant_long = pollutant_avg.melt(
            id_vars="city",
            var_name="pollutant",
            value_name="average_value",
        )

        pollutant_fig = px.bar(
            pollutant_long,
            x="city",
            y="average_value",
            color="pollutant",
            barmode="group",
            title="Average Pollutant Levels by City",
        )
        pollutant_fig = style_fig(pollutant_fig, height=520)

        corr_columns = ["us_aqi"] + available_pollutants
        corr_df = df[corr_columns].corr().round(2)

        corr_fig = px.imshow(
            corr_df,
            text_auto=True,
            title="AQI and Pollutant Correlation Heatmap",
            aspect="auto",
            color_continuous_scale="RdBu_r",
        )
        corr_fig = style_fig(corr_fig, height=520)

        sections.append(
            f"""
            <div class="card">
                <h2>Average Pollutant Levels</h2>
                {plot_to_html(pollutant_fig)}
            </div>
            """
        )

        sections.append(
            f"""
            <div class="card">
                <h2>Pollutant Correlation</h2>
                {plot_to_html(corr_fig)}
            </div>
            """
        )

    if available_targets:
        target_df = df[available_targets].copy()

        target_long = target_df.melt(
            var_name="forecast_target",
            value_name="aqi_value",
        ).dropna()

        target_fig = px.histogram(
            target_long,
            x="aqi_value",
            color="forecast_target",
            nbins=40,
            title="Forecast Target Distribution",
        )
        target_fig = style_fig(target_fig, height=500)

        sections.append(
            f"""
            <div class="card">
                <h2>Forecast Target Distribution</h2>
                {plot_to_html(target_fig)}
            </div>
            """
        )

    missing_summary = (
        df.isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_values"})
    )

    missing_summary["missing_percent"] = (
        (missing_summary["missing_values"] / len(df)) * 100
    ).round(2)

    missing_summary = missing_summary[missing_summary["missing_values"] > 0].sort_values(
        "missing_values",
        ascending=False,
    )

    if missing_summary.empty:
        missing_html = "<p>No missing values found.</p>"
    else:
        missing_html = missing_summary.head(40).to_html(index=False)

    sections.append(
        f"""
        <div class="card">
            <h2>Missing Value Summary</h2>
            {missing_html}
        </div>
        """
    )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": data_source,
        "rows": int(total_rows),
        "cities": int(total_cities),
        "time_min": str(min_time),
        "time_max": str(max_time),
        "average_aqi": safe_round(avg_aqi, 3),
        "max_aqi": safe_round(max_aqi, 3),
        "city_summary": city_summary.to_dict(orient="records"),
        "available_pollutants": available_pollutants,
        "available_targets": available_targets,
    }

    safe_summary = make_json_safe(summary)

    EDA_SUMMARY_JSON_PATH.write_text(
        json.dumps(safe_summary, indent=4),
        encoding="utf-8",
    )

    write_html_report(
        title="AQI Exploratory Data Analysis Report",
        sections=sections,
        output_path=EDA_HTML_PATH,
    )

    print(f"EDA HTML report saved to: {EDA_HTML_PATH}")
    print(f"EDA JSON summary saved to: {EDA_SUMMARY_JSON_PATH}")


def get_feature_importance_from_model(model, X_sample, y_sample):
    try:
        estimator = (
            model.named_steps["model"]
            if hasattr(model, "named_steps") and "model" in model.named_steps
            else model
        )

        if hasattr(estimator, "feature_importances_"):
            values = estimator.feature_importances_

            if hasattr(values, "ravel"):
                values = values.ravel()

            return pd.DataFrame(
                {
                    "feature": FEATURE_COLUMNS,
                    "importance": values,
                    "method": "model_feature_importance",
                }
            )

        if hasattr(estimator, "coef_"):
            values = abs(estimator.coef_)

            if hasattr(values, "ravel"):
                values = values.ravel()

            return pd.DataFrame(
                {
                    "feature": FEATURE_COLUMNS,
                    "importance": values,
                    "method": "model_coefficients",
                }
            )

        result = permutation_importance(
            model,
            X_sample,
            y_sample,
            n_repeats=3,
            random_state=42,
            scoring="neg_mean_absolute_error",
        )

        return pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "importance": abs(result.importances_mean),
                "method": "permutation_importance",
            }
        )

    except Exception as error:
        print("Feature importance failed:", error)
        return pd.DataFrame()


def generate_explainability_report(df: pd.DataFrame, model_bundle, metrics):
    print("Generating explainability report...")

    if model_bundle is None:
        print("Skipping explainability report because model bundle is not available.")
        return

    all_importance_frames = []

    for target_key, target_column in TARGET_COLUMNS.items():
        if target_key not in model_bundle.get("models", {}):
            print(f"Skipping {target_key}: model not found in bundle.")
            continue

        if target_column not in df.columns:
            print(f"Skipping {target_key}: target column not found.")
            continue

        sample_df = df.dropna(subset=FEATURE_COLUMNS + [target_column]).copy()

        if len(sample_df) < 50:
            print(f"Skipping {target_key}: not enough clean rows.")
            continue

        sample_df = sample_df.tail(800)

        X_sample = sample_df[FEATURE_COLUMNS]
        y_sample = sample_df[target_column]

        model_info = model_bundle["models"][target_key]
        model = model_info["model"]
        model_name = model_info.get("model_name", "unknown_model")

        importance_df = get_feature_importance_from_model(model, X_sample, y_sample)

        if importance_df.empty:
            continue

        importance_df["target"] = target_key
        importance_df["target_column"] = target_column
        importance_df["model_name"] = model_name
        importance_df = importance_df.sort_values("importance", ascending=False)
        all_importance_frames.append(importance_df)

    if not all_importance_frames:
        print("No explainability data generated.")
        return

    full_importance_df = pd.concat(all_importance_frames, ignore_index=True)
    full_importance_df.to_csv(EXPLAINABILITY_CSV_PATH, index=False)

    sections = []

    sections.append(
        """
        <div class="note">
            This report explains which features influenced the AQI forecasting models the most.
            It uses model-native feature importance where available and permutation importance as fallback.
            SHAP report is generated separately in <b>shap_report.html</b>.
        </div>
        """
    )

    for target_key in full_importance_df["target"].unique():
        target_importance = (
            full_importance_df[full_importance_df["target"] == target_key]
            .sort_values("importance", ascending=False)
            .head(20)
        )

        fig = px.bar(
            target_importance.sort_values("importance", ascending=True),
            x="importance",
            y="feature",
            orientation="h",
            title=f"Top Feature Importance for {target_key}",
            color="importance",
            color_continuous_scale="Blues",
        )
        fig = style_fig(fig, height=620)

        table_df = target_importance[
            ["target", "model_name", "feature", "importance", "method"]
        ].copy()

        table_df["importance"] = table_df["importance"].round(6)

        sections.append(
            f"""
            <div class="card">
                <h2>Explainability: {target_key}</h2>
                {plot_to_html(fig)}
                {table_df.to_html(index=False)}
            </div>
            """
        )

    if metrics:
        safe_metrics = make_json_safe(metrics)

        sections.append(
            f"""
            <div class="card">
                <h2>Model Metrics Snapshot</h2>
                <pre>{json.dumps(safe_metrics, indent=4)}</pre>
            </div>
            """
        )

    write_html_report(
        title="AQI Model Explainability Report",
        sections=sections,
        output_path=EXPLAINABILITY_HTML_PATH,
    )

    print(f"Explainability HTML report saved to: {EXPLAINABILITY_HTML_PATH}")
    print(f"Feature importance CSV saved to: {EXPLAINABILITY_CSV_PATH}")


def calculate_shap_importance(model, X_background: pd.DataFrame, X_explain: pd.DataFrame):
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed.")

    masker = shap.maskers.Independent(X_background, max_samples=min(60, len(X_background)))

    explainer = shap.Explainer(
        model.predict,
        masker,
        feature_names=list(X_explain.columns),
    )

    shap_values = explainer(X_explain)

    values = shap_values.values

    if isinstance(values, list):
        values = values[0]

    values = np.asarray(values)

    if values.ndim == 3:
        values = values[:, :, 0]

    mean_abs_values = np.abs(values).mean(axis=0)

    return pd.DataFrame(
        {
            "feature": list(X_explain.columns),
            "mean_abs_shap": mean_abs_values,
        }
    ).sort_values("mean_abs_shap", ascending=False)


def generate_shap_report(df: pd.DataFrame, model_bundle):
    print("Generating SHAP report...")

    if not SHAP_AVAILABLE:
        warning_section = """
        <div class="warning">
            SHAP package is not installed in this environment.
            Install it using <b>pip install shap==0.46.0</b> and rerun the report.
        </div>
        """

        write_html_report(
            title="AQI SHAP Explainability Report",
            sections=[warning_section],
            output_path=SHAP_HTML_PATH,
        )

        print("SHAP is not available. SHAP report generated with warning only.")
        return

    if model_bundle is None:
        warning_section = """
        <div class="warning">
            Model bundle is not available, so SHAP values could not be calculated.
            Run the training pipeline first.
        </div>
        """

        write_html_report(
            title="AQI SHAP Explainability Report",
            sections=[warning_section],
            output_path=SHAP_HTML_PATH,
        )

        print("Skipping SHAP report because model bundle is not available.")
        return

    all_shap_frames = []
    sections = []

    sections.append(
        """
        <div class="note">
            SHAP explains how much each feature contributes to the model prediction.
            Higher mean absolute SHAP values mean the feature has stronger influence on the AQI forecast.
            To keep runtime safe in local and GitHub Actions environments, SHAP is calculated on a recent sample of rows.
        </div>
        """
    )

    for target_key, target_column in TARGET_COLUMNS.items():
        if target_key not in model_bundle.get("models", {}):
            print(f"Skipping SHAP for {target_key}: model not found.")
            continue

        if target_column not in df.columns:
            print(f"Skipping SHAP for {target_key}: target column not found.")
            continue

        clean_df = df.dropna(subset=FEATURE_COLUMNS + [target_column]).copy()

        if len(clean_df) < 80:
            print(f"Skipping SHAP for {target_key}: not enough clean rows.")
            continue

        clean_df = clean_df.tail(500)

        background_size = min(60, len(clean_df))
        explain_size = min(120, len(clean_df))

        background_df = clean_df.sample(
            n=background_size,
            random_state=42,
        )

        explain_df = clean_df.tail(explain_size)

        X_background = background_df[FEATURE_COLUMNS]
        X_explain = explain_df[FEATURE_COLUMNS]

        model_info = model_bundle["models"][target_key]
        model = model_info["model"]
        model_name = model_info.get("model_name", "unknown_model")

        try:
            shap_importance = calculate_shap_importance(
                model=model,
                X_background=X_background,
                X_explain=X_explain,
            )

            shap_importance["target"] = target_key
            shap_importance["target_column"] = target_column
            shap_importance["model_name"] = model_name
            shap_importance["rows_explained"] = len(X_explain)
            shap_importance["background_rows"] = len(X_background)

            all_shap_frames.append(shap_importance)

            top_shap = shap_importance.head(20).copy()

            fig = px.bar(
                top_shap.sort_values("mean_abs_shap", ascending=True),
                x="mean_abs_shap",
                y="feature",
                orientation="h",
                title=f"SHAP Feature Impact for {target_key}",
                color="mean_abs_shap",
                color_continuous_scale="Viridis",
            )
            fig = style_fig(fig, height=620)

            table_df = top_shap[
                [
                    "target",
                    "model_name",
                    "feature",
                    "mean_abs_shap",
                    "rows_explained",
                    "background_rows",
                ]
            ].copy()

            table_df["mean_abs_shap"] = table_df["mean_abs_shap"].round(6)

            sections.append(
                f"""
                <div class="card">
                    <h2>SHAP Explanation: {target_key}</h2>
                    {plot_to_html(fig)}
                    {table_df.to_html(index=False)}
                </div>
                """
            )

            print(f"SHAP completed for {target_key}.")

        except Exception as error:
            error_message = str(error)

            sections.append(
                f"""
                <div class="warning">
                    <b>SHAP failed for {target_key}</b><br>
                    Model: {model_name}<br>
                    Error: {error_message}
                </div>
                """
            )

            print(f"SHAP failed for {target_key}: {error}")

    if all_shap_frames:
        full_shap_df = pd.concat(all_shap_frames, ignore_index=True)
        full_shap_df.to_csv(SHAP_CSV_PATH, index=False)
        print(f"SHAP CSV saved to: {SHAP_CSV_PATH}")
    else:
        sections.append(
            """
            <div class="warning">
                No SHAP values were generated. Check model availability, clean rows, and SHAP installation.
            </div>
            """
        )

    write_html_report(
        title="AQI SHAP Explainability Report",
        sections=sections,
        output_path=SHAP_HTML_PATH,
    )

    print(f"SHAP HTML report saved to: {SHAP_HTML_PATH}")


def main():
    ensure_directories()

    df, data_source = load_feature_data()
    model_bundle = load_model_bundle()
    metrics = load_metrics()

    generate_eda_report(df=df, data_source=data_source)

    generate_explainability_report(
        df=df,
        model_bundle=model_bundle,
        metrics=metrics,
    )

    generate_shap_report(
        df=df,
        model_bundle=model_bundle,
    )

    print("EDA, explainability, and SHAP reports completed.")


if __name__ == "__main__":
    main()