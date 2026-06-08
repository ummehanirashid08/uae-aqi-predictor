import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from config import FEATURE_STORE_PATH, MODEL_PATH, METRICS_PATH, get_bool_setting, get_setting
from hopsworks_store import load_features_from_hopsworks


os.environ["LOKY_MAX_CPU_COUNT"] = "4"


# =========================
# CONFIG / CONSTANTS
# =========================
CITY_MAP = {
    "24.1302,55.8023": "Al Ain",
    "24.4539,54.3773": "Abu Dhabi",
    "25.2048,55.2708": "Dubai",
    "25.3463,55.4209": "Sharjah",
    "25.4052,55.5136": "Ajman",
    "25.8007,55.9762": "Ras Al Khaimah",
}

CITY_VISUALS = {
    "Dubai": {"emoji": "🗼", "landmark": "Burj Khalifa vibe"},
    "Abu Dhabi": {"emoji": "🕌", "landmark": "Sheikh Zayed vibe"},
    "Sharjah": {"emoji": "📚", "landmark": "Cultural capital vibe"},
    "Ajman": {"emoji": "🌊", "landmark": "Coastal city vibe"},
    "Al Ain": {"emoji": "🌴", "landmark": "Garden city vibe"},
    "Ras Al Khaimah": {"emoji": "⛰️", "landmark": "Mountain city vibe"},
}

DAY_OPTIONS = {
    "Day 1 Forecast": "day_1",
    "Day 2 Forecast": "day_2",
    "Day 3 Forecast": "day_3",
    "3-Day Average": "next_72h_avg",
}

DAY_ICONS = {
    "Day 1 Forecast": "📍",
    "Day 2 Forecast": "📅",
    "Day 3 Forecast": "🧭",
    "3-Day Average": "📊",
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

POLLUTANT_SHORT = {
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "carbon_monoxide": "CO",
    "nitrogen_dioxide": "NO₂",
    "sulphur_dioxide": "SO₂",
    "ozone": "O₃",
    "dust": "Dust",
}

POLLUTANT_NAMES = {
    "pm2_5": "Particulate Matter",
    "pm10": "Particulate Matter",
    "carbon_monoxide": "Carbon Monoxide",
    "nitrogen_dioxide": "Nitrogen Dioxide",
    "sulphur_dioxide": "Sulphur Dioxide",
    "ozone": "Ozone",
    "dust": "Dust",
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


# =========================
# PAGE / CSS
# =========================
def setup_page():
    st.set_page_config(
        page_title="UAE AQI Predictor",
        page_icon="🌤️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 8% 2%, rgba(250, 204, 21, 0.22), transparent 28%),
        radial-gradient(circle at 92% 4%, rgba(59, 130, 246, 0.12), transparent 32%),
        linear-gradient(135deg, #f8fafc 0%, #eef4ff 100%);
}

.block-container {
    padding-top: 2.2rem !important;
    padding-bottom: 4rem !important;
    max-width: 1400px;
}

section[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.94);
    border-right: 1px solid rgba(226, 232, 240, 0.95);
}

section[data-testid="stSidebar"] h1 {
    color: #0f172a;
    font-weight: 900;
    letter-spacing: -0.8px;
    line-height: 1.12;
}

div[data-testid="stRadio"] label {
    color: #334155 !important;
    font-weight: 650;
}

div[data-testid="stAlert"] {
    border-radius: 18px;
}

.brand-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 22px;
    margin-bottom: 26px;
    padding-top: 4px;
}

.brand-left {
    display: flex;
    align-items: center;
    gap: 14px;
}

.brand-logo {
    width: 58px;
    height: 58px;
    border-radius: 18px;
    background: linear-gradient(135deg, #60a5fa, #2563eb);
    color: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 900;
    font-size: 21px;
    box-shadow: 0 16px 34px rgba(37, 99, 235, 0.24);
}

.brand-title {
    font-size: 36px;
    color: #0f172a;
    font-weight: 900;
    letter-spacing: -1.1px;
}

.brand-pills {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    justify-content: flex-end;
}

.brand-pill {
    padding: 12px 17px;
    border-radius: 999px;
    border: 1px solid #d4b82f;
    color: #0f172a;
    background: rgba(255, 255, 255, 0.86);
    font-size: 13px;
    font-weight: 900;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
}

.hero-yellow {
    background: linear-gradient(135deg, #fff8d8 0%, #facc15 100%);
    border: 1px solid #f3d56b;
    border-radius: 34px;
    padding: 44px 46px;
    margin-bottom: 30px;
    box-shadow: 0 28px 68px rgba(15, 23, 42, 0.10);
}

.live-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 18px;
    border-radius: 999px;
    color: #ffffff;
    background: #ef4444;
    font-size: 12px;
    font-weight: 900;
    margin-bottom: 24px;
    box-shadow: 0 12px 24px rgba(239, 68, 68, 0.22);
}

.city-vibe {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 16px;
    padding: 10px 16px;
    border-radius: 999px;
    background: rgba(255,255,255,0.55);
    border: 1px solid rgba(255,255,255,0.65);
    color: #0f172a;
    font-size: 13px;
    font-weight: 900;
}

.hero-title {
    font-size: 45px;
    font-weight: 900;
    letter-spacing: -1.5px;
    color: #0f172a;
    margin-bottom: 14px;
    line-height: 1.05;
}

.hero-subtitle {
    font-size: 17px;
    line-height: 1.7;
    color: #334155;
    font-weight: 800;
    max-width: 920px;
}

.top-card {
    background: rgba(255, 253, 242, 0.98);
    border: 1px solid #f3e8a2;
    border-radius: 30px;
    padding: 32px;
    min-height: 310px;
    box-shadow: 0 24px 58px rgba(15, 23, 42, 0.09);
    overflow: hidden;
}

.top-card-label {
    font-size: 15px;
    color: #334155;
    font-weight: 900;
    margin-bottom: 18px;
}

.live-number {
    font-size: 74px;
    line-height: 0.95;
    color: #ca8a04;
    font-weight: 900;
    letter-spacing: -2.4px;
    margin-bottom: 22px;
}

.forecast-status {
    display: flex;
    align-items: center;
    gap: 14px;
    color: #f97316;
    font-size: 34px;
    font-weight: 900;
    letter-spacing: -1px;
    line-height: 1.28;
    margin-bottom: 24px;
}

.forecast-number-label {
    color: #64748b;
    font-size: 13px;
    font-weight: 900;
    margin-bottom: 8px;
}

.forecast-number {
    color: #ca8a04;
    font-size: 68px;
    line-height: 0.95;
    font-weight: 900;
    letter-spacing: 0;
    margin-bottom: 18px;
}

.forecast-category-badge {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    max-width: 100%;
    padding: 10px 15px;
    border-radius: 999px;
    background: rgba(255, 247, 214, 0.9);
    border: 1px solid #f3d56b;
    color: #f97316;
    font-size: 22px;
    font-weight: 900;
    line-height: 1.25;
    margin-bottom: 20px;
}

.forecast-category-badge span:last-child {
    overflow-wrap: anywhere;
}

.forecast-category-badge .status-dot {
    width: 18px;
    height: 18px;
    min-width: 18px;
    box-shadow: 0 8px 16px rgba(239, 68, 68, 0.18);
}

.forecast-details {
    color: #475569;
    font-weight: 850;
    line-height: 1.8;
    font-size: 15px;
}

.status-dot {
    width: 38px;
    height: 38px;
    min-width: 38px;
    border-radius: 999px;
    background: linear-gradient(135deg, #fb923c, #ef4444);
    box-shadow: 0 12px 24px rgba(239, 68, 68, 0.24);
}

.top-card-desc {
    color: #475569;
    font-weight: 800;
    line-height: 1.72;
    font-size: 15px;
}

.mini-row {
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
    margin-top: 24px;
    color: #334155;
    font-size: 15px;
    font-weight: 900;
}

.weather-icon {
    font-size: 58px;
    margin-bottom: 18px;
}

.weather-temp {
    font-size: 62px;
    color: #0f172a;
    font-weight: 900;
    letter-spacing: -1.8px;
    margin-bottom: 10px;
}

.weather-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-top: 24px;
}

.weather-chip {
    border: 1px solid #e5e7eb;
    background: rgba(255, 255, 255, 0.82);
    border-radius: 18px;
    padding: 15px 12px;
    text-align: center;
}

.weather-chip-label {
    color: #64748b;
    font-size: 12px;
    font-weight: 900;
    margin-bottom: 7px;
}

.weather-chip-value {
    color: #0f172a;
    font-size: 16px;
    font-weight: 900;
}

.source-strip {
    background: #fffdf2;
    border: 1px solid #f3e8a2;
    border-radius: 22px;
    padding: 20px 24px;
    margin: 24px 0 26px 0;
    color: #334155;
    font-weight: 800;
    line-height: 1.82;
    box-shadow: 0 14px 34px rgba(15, 23, 42, 0.05);
}

.section-title {
    font-size: 32px;
    font-weight: 900;
    letter-spacing: -0.9px;
    color: #0f172a;
    margin: 50px 0 22px 0;
}

.metric-card {
    background: rgba(255, 255, 255, 0.94);
    border: 1px solid #e5e7eb;
    border-radius: 26px;
    padding: 25px;
    min-height: 178px;
    box-shadow: 0 16px 36px rgba(15, 23, 42, 0.06);
}

.metric-card.active {
    border: 2px solid #ef4444;
    background: rgba(255, 250, 245, 0.96);
}

.metric-title {
    color: #475569;
    font-size: 15px;
    font-weight: 900;
    margin-bottom: 16px;
}

.metric-value {
    color: #0f172a;
    font-size: 40px;
    font-weight: 900;
    letter-spacing: -1px;
    margin-bottom: 14px;
    line-height: 1.1;
}

.metric-desc {
    color: #334155;
    font-size: 14px;
    font-weight: 900;
    line-height: 1.5;
}

.aqi-scale-card {
    background: #fffdf2;
    border: 1px solid #f3e8a2;
    border-radius: 28px;
    padding: 30px;
    box-shadow: 0 20px 48px rgba(15, 23, 42, 0.07);
}

.scale-track {
    height: 13px;
    border-radius: 999px;
    background: linear-gradient(90deg, #22c55e 0%, #22c55e 10%, #eab308 10%, #eab308 20%, #f97316 20%, #f97316 30%, #ef4444 30%, #ef4444 40%, #8b5cf6 40%, #8b5cf6 60%, #111827 60%, #111827 100%);
    position: relative;
    margin-top: 42px;
}

.scale-marker-live {
    position: absolute;
    top: -26px;
    width: 3px;
    height: 42px;
    background: #2563eb;
    border-radius: 99px;
}

.scale-marker-forecast {
    position: absolute;
    top: -26px;
    width: 3px;
    height: 42px;
    background: #111827;
    border-radius: 99px;
}

.scale-badge {
    position: absolute;
    top: -54px;
    transform: translateX(-50%);
    background: #111827;
    color: white;
    font-size: 11px;
    font-weight: 900;
    padding: 7px 11px;
    border-radius: 999px;
    white-space: nowrap;
}

.scale-labels {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    margin-top: 22px;
    color: #334155;
    font-size: 12px;
    font-weight: 900;
    text-align: center;
}

.scale-numbers {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    margin-top: 15px;
    color: #64748b;
    font-size: 12px;
    font-weight: 800;
    text-align: center;
}

.premium-card {
    background: rgba(255, 253, 242, 0.98);
    border: 1px solid #f3e8a2;
    border-radius: 30px;
    padding: 30px;
    box-shadow: 0 22px 56px rgba(15, 23, 42, 0.08);
    margin: 26px 0;
}

.premium-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 18px;
    margin-bottom: 24px;
}

.premium-title {
    color: #0f172a;
    font-size: 31px;
    font-weight: 900;
    letter-spacing: -0.9px;
    line-height: 1.1;
}

.premium-subtitle {
    color: #64748b;
    font-size: 15px;
    font-weight: 800;
    margin-top: 9px;
    line-height: 1.6;
}

.premium-chip {
    padding: 12px 16px;
    border-radius: 999px;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    color: #2563eb;
    font-size: 13px;
    font-weight: 900;
    white-space: nowrap;
}

.aqiin-mini-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin: 20px 0 10px 0;
}

.aqiin-mini-card {
    background: rgba(255, 255, 255, 0.88);
    border: 1px solid #e5e7eb;
    border-radius: 22px;
    padding: 20px;
    min-height: 128px;
    box-shadow: 0 14px 34px rgba(15, 23, 42, 0.05);
}

.aqiin-mini-label {
    color: #64748b;
    font-size: 13px;
    font-weight: 900;
    margin-bottom: 12px;
}

.aqiin-mini-value {
    color: #0f172a;
    font-size: 32px;
    font-weight: 900;
    line-height: 1.1;
}

.aqiin-mini-status {
    margin-top: 12px;
    font-size: 13px;
    font-weight: 900;
    color: #475569;
}

.trend-intro-card {
    background: rgba(255, 253, 242, 0.98);
    border: 1px solid #f3e8a2;
    border-radius: 28px;
    padding: 26px 34px;
    box-shadow: 0 18px 42px rgba(15, 23, 42, 0.06);
    margin: 18px 0 20px 0;
}

.trend-chart-shell {
    background: #fffdf2;
    border: 1px solid #f3e8a2;
    border-radius: 30px;
    padding: 24px;
    box-shadow: 0 22px 54px rgba(15, 23, 42, 0.08);
    margin-top: 0;
}

.trend-stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-top: 18px;
}

.trend-stat {
    background: rgba(255, 255, 255, 0.84);
    border: 1px solid #e5e7eb;
    border-radius: 20px;
    padding: 18px;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
}

.trend-stat-label {
    color: #64748b;
    font-size: 13px;
    font-weight: 900;
    margin-bottom: 8px;
}

.trend-stat-value {
    color: #0f172a;
    font-size: 22px;
    font-weight: 900;
}

.recommend-card {
    background: #fffdf2;
    border: 1px solid #f3e8a2;
    border-radius: 24px;
    padding: 24px;
    min-height: 160px;
    box-shadow: 0 14px 34px rgba(15, 23, 42, 0.05);
}

.recommend-card h4 {
    margin: 0 0 14px 0;
    font-size: 18px;
    font-weight: 900;
    color: #0f172a;
}

.recommend-card p {
    margin: 0;
    color: #475569;
    line-height: 1.75;
    font-size: 15px;
    font-weight: 800;
}

.pollutant-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 18px;
}

.pollutant-card {
    background: rgba(248, 250, 252, 0.95);
    border: 1px solid #e5e7eb;
    border-left: 6px solid #2563eb;
    border-radius: 22px;
    padding: 20px;
    min-height: 104px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.04);
}

.pollutant-left {
    display: flex;
    align-items: center;
    gap: 13px;
}

.pollutant-icon {
    width: 46px;
    height: 46px;
    min-width: 46px;
    border-radius: 15px;
    background: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    box-shadow: 0 10px 22px rgba(15, 23, 42, 0.05);
}

.pollutant-name {
    font-size: 15px;
    font-weight: 900;
    color: #0f172a;
}

.pollutant-short {
    font-size: 13px;
    color: #64748b;
    font-weight: 800;
}

.pollutant-right {
    text-align: right;
}

.pollutant-value {
    color: #0f172a;
    font-size: 23px;
    font-weight: 900;
}

.pollutant-unit {
    color: #64748b;
    font-size: 12px;
    font-weight: 800;
}

.ranking-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 18px;
}

.rank-card {
    background: rgba(255,255,255,0.9);
    border: 1px solid #e5e7eb;
    border-radius: 24px;
    padding: 22px;
    box-shadow: 0 14px 34px rgba(15,23,42,0.05);
}

.rank-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 14px;
}

.rank-city {
    font-size: 20px;
    font-weight: 900;
    color: #0f172a;
}

.rank-badge {
    padding: 7px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 900;
    background: #f8fafc;
    color: #334155;
    border: 1px solid #e5e7eb;
}

.rank-aqi {
    font-size: 42px;
    font-weight: 900;
    letter-spacing: -1px;
}

.rank-status {
    color: #475569;
    font-size: 14px;
    font-weight: 850;
    margin-top: 8px;
}

.rank-landmark {
    color: #94a3b8;
    font-size: 12px;
    font-weight: 800;
    margin-top: 6px;
}

.why-main {
    background: rgba(255, 255, 255, 0.88);
    border: 1px solid #e5e7eb;
    border-radius: 24px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 14px 32px rgba(15, 23, 42, 0.04);
}

.why-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 18px;
}

.why-card {
    background: rgba(255, 255, 255, 0.86);
    border: 1px solid #e5e7eb;
    border-radius: 24px;
    padding: 24px;
    min-height: 170px;
    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.04);
}

.why-icon {
    width: 44px;
    height: 44px;
    border-radius: 15px;
    background: #eff6ff;
    color: #2563eb;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    margin-bottom: 14px;
}

.why-title {
    color: #0f172a;
    font-size: 18px;
    font-weight: 900;
    margin-bottom: 9px;
}

.why-text {
    color: #334155;
    font-size: 14px;
    font-weight: 750;
    line-height: 1.75;
}

.why-tag {
    display: inline-block;
    margin-top: 15px;
    padding: 8px 11px;
    border-radius: 999px;
    background: #fff1f2;
    color: #be123c;
    font-weight: 900;
    font-size: 12px;
}

.model-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 18px;
    margin-bottom: 22px;
}

.model-card {
    background: rgba(255, 255, 255, 0.88);
    border: 1px solid #e5e7eb;
    border-radius: 25px;
    padding: 23px;
    min-height: 185px;
    box-shadow: 0 14px 34px rgba(15, 23, 42, 0.04);
}

.model-icon {
    width: 50px;
    height: 50px;
    border-radius: 16px;
    background: #eff6ff;
    color: #2563eb;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    margin-bottom: 16px;
}

.model-label {
    color: #64748b;
    font-weight: 900;
    font-size: 13px;
    margin-bottom: 8px;
}

.model-value {
    color: #0f172a;
    font-size: 27px;
    font-weight: 900;
    letter-spacing: -0.8px;
    line-height: 1.2;
}

.model-desc {
    color: #475569;
    font-size: 14px;
    font-weight: 750;
    line-height: 1.55;
    margin-top: 12px;
}

.model-explain-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
}

.model-explain-box {
    background: #fff7d6;
    border: 1px solid #f3d56b;
    border-radius: 18px;
    padding: 17px;
    color: #334155;
    font-weight: 750;
    line-height: 1.75;
}

.tech-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
}

.tech-box {
    background: rgba(255, 255, 255, 0.88);
    border: 1px solid #e5e7eb;
    border-radius: 20px;
    padding: 18px;
}

.tech-label {
    color: #64748b;
    font-size: 12px;
    font-weight: 900;
    margin-bottom: 8px;
}

.tech-value {
    color: #0f172a;
    font-size: 18px;
    font-weight: 900;
    line-height: 1.35;
}

.footer-premium {
    margin-top: 58px;
    background: linear-gradient(135deg, #fffaf0 0%, #eef4ff 100%);
    border: 1px solid #f3e8a2;
    border-radius: 32px;
    padding: 30px;
    box-shadow: 0 22px 58px rgba(15,23,42,0.08);
}

.footer-grid {
    display: grid;
    grid-template-columns: 1.4fr 1fr 1fr;
    gap: 18px;
}

.footer-box {
    background: rgba(255,255,255,0.75);
    border: 1px solid #e5e7eb;
    border-radius: 22px;
    padding: 22px;
}

.footer-main-title {
    font-size: 26px;
    font-weight: 900;
    color: #0f172a;
    letter-spacing: -0.8px;
    margin-bottom: 10px;
}

.footer-main-subtitle {
    font-size: 14px;
    line-height: 1.8;
    font-weight: 800;
    color: #475569;
}

.footer-title {
    font-size: 15px;
    font-weight: 900;
    color: #0f172a;
    margin-bottom: 14px;
}

.footer-item {
    color: #475569;
    font-size: 14px;
    line-height: 1.8;
    font-weight: 800;
}

.footer-bottom {
    border-top: 1px solid #e5e7eb;
    margin-top: 20px;
    padding-top: 18px;
    display: flex;
    justify-content: space-between;
    gap: 18px;
    flex-wrap: wrap;
    color: #64748b;
    font-size: 13px;
    font-weight: 800;
}

div[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
}

.js-plotly-plot {
    border-radius: 24px;
    overflow: hidden;
}

@media (max-width: 1100px) {
    .model-grid,
    .why-grid,
    .tech-grid,
    .trend-stat-grid,
    .aqiin-mini-grid,
    .ranking-grid,
    .footer-grid {
        grid-template-columns: repeat(2, 1fr);
    }

    .pollutant-grid {
        grid-template-columns: repeat(2, 1fr);
    }

    .model-explain-row {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 720px) {
    .brand-row {
        flex-direction: column;
        align-items: flex-start;
    }

    .model-grid,
    .why-grid,
    .pollutant-grid,
    .tech-grid,
    .trend-stat-grid,
    .aqiin-mini-grid,
    .ranking-grid,
    .footer-grid {
        grid-template-columns: 1fr;
    }

    .hero-title {
        font-size: 32px;
    }

    .live-number {
        font-size: 52px;
    }

    .forecast-status {
        font-size: 28px;
    }

    .forecast-number {
        font-size: 56px;
    }

    .forecast-category-badge {
        font-size: 20px;
        border-radius: 18px;
    }
}
</style>
        """,
        unsafe_allow_html=True,
    )


# =========================
# HELPERS
# =========================
def clean_model_name(model_name):
    names = {
        "hist_gradient_boosting": "Histogram Gradient Boosting",
        "gradient_boosting": "Gradient Boosting",
        "random_forest": "Random Forest",
        "extra_trees": "Extra Trees",
        "xgboost": "XGBoost",
        "ridge": "Ridge Regression",
        "elastic_net": "Elastic Net",
        "baseline_current_aqi": "Baseline AQI",
    }
    return names.get(str(model_name), str(model_name).replace("_", " ").title())


def clean_target_name(target_key):
    names = {
        "day_1": "Day 1 Forecast",
        "day_2": "Day 2 Forecast",
        "day_3": "Day 3 Forecast",
        "next_72h_avg": "3-Day Average",
    }
    return names.get(target_key, str(target_key).replace("_", " ").title())


def get_aqi_category(aqi):
    try:
        aqi = float(aqi)
    except Exception:
        return "Unknown", "AQI is not available.", "⚪", "#64748b"

    if aqi <= 50:
        return "Good", "Air quality is satisfactory.", "🟢", "#22c55e"
    if aqi <= 100:
        return "Moderate", "Air quality is acceptable.", "🟡", "#eab308"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups", "Sensitive people should reduce outdoor activity.", "🟠", "#f97316"
    if aqi <= 200:
        return "Unhealthy", "Everyone may begin to experience health effects. Outdoor activity should be reduced.", "🔴", "#ef4444"
    if aqi <= 300:
        return "Very Unhealthy", "Health alert. Outdoor activity should be avoided.", "🟣", "#8b5cf6"
    return "Hazardous", "Emergency conditions. Stay indoors.", "⚫", "#111827"


def get_accuracy_label(r2_value):
    if r2_value is None:
        return "Not Available", "Model score is not available.", "#64748b", "⚪"

    try:
        r2_value = float(r2_value)
    except Exception:
        return "Not Available", "Model score is not available.", "#64748b", "⚪"

    if r2_value >= 0.80:
        return "Strong", "The model explains most AQI changes well.", "#22c55e", "✅"
    if r2_value >= 0.60:
        return "Good", "The model captures AQI patterns well.", "#2563eb", "👍"
    if r2_value >= 0.40:
        return "Moderate", "The model gives a useful estimate but may vary.", "#eab308", "⚠️"
    return "Needs Improvement", "The model has limited accuracy for this forecast.", "#ef4444", "🔎"


def get_pollutant_color(pollutant, value):
    try:
        value = float(value)
    except Exception:
        return "#2563eb"

    if pollutant == "pm2_5":
        if value <= 12:
            return "#22c55e"
        if value <= 35:
            return "#eab308"
        if value <= 55:
            return "#f97316"
        return "#ef4444"

    if pollutant == "pm10":
        if value <= 54:
            return "#22c55e"
        if value <= 154:
            return "#eab308"
        if value <= 254:
            return "#f97316"
        return "#ef4444"

    if value <= 50:
        return "#22c55e"
    if value <= 150:
        return "#eab308"
    if value <= 300:
        return "#f97316"
    return "#ef4444"


def get_health_recommendations(aqi):
    try:
        aqi = float(aqi)
    except Exception:
        return [
            "📡 AQI is not available right now.",
            "🧪 Check feature pipeline data.",
            "🔁 Run training again if needed.",
        ]

    if aqi <= 50:
        return [
            "🌿 Outdoor activity is generally safe.",
            "📈 Keep monitoring AQI through the day.",
            "😌 No special precautions needed.",
        ]
    if aqi <= 100:
        return [
            "🚶 Most people can continue normal activity.",
            "🤧 Sensitive people should monitor symptoms.",
            "🏠 Keep windows closed if AQI worsens.",
        ]
    if aqi <= 150:
        return [
            "😷 Sensitive groups should reduce prolonged outdoor activity.",
            "🫁 People with asthma should carry medication.",
            "🛡️ Consider wearing a mask outdoors.",
        ]
    if aqi <= 200:
        return [
            "🏃 Limit outdoor exercise.",
            "👶 Children and elderly people should stay indoors.",
            "🚪 Keep doors and windows closed.",
        ]
    if aqi <= 300:
        return [
            "🚫 Avoid outdoor activity unless necessary.",
            "🌬️ Use air purification indoors.",
            "⚠️ Sensitive groups should avoid exposure.",
        ]
    return [
        "⛔ Stay indoors.",
        "🌬️ Use air purifiers.",
        "📢 Follow official health alerts.",
    ]


def get_city_visual(city_name):
    return CITY_VISUALS.get(city_name, {"emoji": "📍", "landmark": "UAE city vibe"})


# =========================
# LIVE DATA
# =========================
@st.cache_data(ttl=15 * 60, show_spinner=False)
def fetch_live_current_aqi(latitude, longitude, timezone="Asia/Dubai"):
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": round(float(latitude), 4),
        "longitude": round(float(longitude), 4),
        "timezone": timezone,
        "current": "us_aqi,pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,dust",
    }

    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code in [429, 502, 503, 504]:
                time.sleep(2 + attempt * 2)
                continue
            response.raise_for_status()
            return response.json().get("current")
        except Exception:
            time.sleep(2 + attempt * 2)

    return None


@st.cache_data(ttl=15 * 60, show_spinner=False)
def fetch_live_current_weather(latitude, longitude, timezone="Asia/Dubai"):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": round(float(latitude), 4),
        "longitude": round(float(longitude), 4),
        "timezone": timezone,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation,weather_code",
    }

    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code in [429, 502, 503, 504]:
                time.sleep(2 + attempt * 2)
                continue
            response.raise_for_status()
            return response.json().get("current")
        except Exception:
            time.sleep(2 + attempt * 2)

    return None


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_future_weather(latitude, longitude, timezone="Asia/Dubai"):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": round(float(latitude), 4),
        "longitude": round(float(longitude), 4),
        "timezone": timezone,
        "forecast_days": 4,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,surface_pressure,precipitation",
    }

    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=40)
            if response.status_code in [429, 502, 503, 504]:
                time.sleep(2 + attempt * 2)
                continue
            response.raise_for_status()
            data = response.json()
            if "hourly" not in data:
                return None
            future_df = pd.DataFrame(data["hourly"])
            future_df["time"] = pd.to_datetime(future_df["time"], errors="coerce")
            future_df = future_df.dropna(subset=["time"])
            return future_df
        except Exception:
            time.sleep(2 + attempt * 2)

    return None


# =========================
# DATA PREP
# =========================
def safe_mean(series):
    if series is None or len(series) == 0:
        return None
    value = series.mean()
    return None if pd.isna(value) else float(value)


def safe_sum(series):
    if series is None or len(series) == 0:
        return None
    value = series.sum()
    return None if pd.isna(value) else float(value)


def safe_max(series):
    if series is None or len(series) == 0:
        return None
    value = series.max()
    return None if pd.isna(value) else float(value)


def update_row_with_future_weather(latest_row, future_df):
    updated_row = latest_row.copy()

    if future_df is None or future_df.empty or len(future_df) < 72:
        return updated_row, False

    future_df = future_df.sort_values("time").reset_index(drop=True)

    windows = {
        "day1": future_df.iloc[0:24],
        "day2": future_df.iloc[24:48],
        "day3": future_df.iloc[48:72],
    }

    mapping = {
        "temperature_2m": "temp",
        "relative_humidity_2m": "humidity",
        "wind_speed_10m": "wind",
        "wind_direction_10m": "wind_direction",
        "surface_pressure": "pressure",
    }

    applied = 0

    for day_name, window_df in windows.items():
        for source_col, short_name in mapping.items():
            feature_name = f"future_{short_name}_{day_name}_avg"
            if source_col in window_df.columns and feature_name in updated_row.index:
                value = safe_mean(window_df[source_col])
                if value is not None:
                    updated_row[feature_name] = value
                    applied += 1

        precip_feature = f"future_precip_{day_name}_sum"
        if "precipitation" in window_df.columns and precip_feature in updated_row.index:
            value = safe_sum(window_df["precipitation"])
            if value is not None:
                updated_row[precip_feature] = value
                applied += 1

        wind_max_feature = f"future_wind_{day_name}_max"
        if "wind_speed_10m" in window_df.columns and wind_max_feature in updated_row.index:
            value = safe_max(window_df["wind_speed_10m"])
            if value is not None:
                updated_row[wind_max_feature] = value
                applied += 1

    return updated_row, applied > 0


def normalize_columns(df):
    df = df.copy()
    df.columns = [
        str(column).strip().lower().replace("-", "_").replace(".", "_")
        for column in df.columns
    ]
    return df


@st.cache_data(ttl=300, show_spinner=False)
def load_feature_store():
    cloud_df, cloud_source = load_features_from_hopsworks()

    if cloud_df is not None and not cloud_df.empty:
        df = normalize_columns(cloud_df)
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"])
        df = df.sort_values("time").reset_index(drop=True)
        return df, "Hopsworks Feature Store"

    if not FEATURE_STORE_PATH.exists():
        st.error("Feature store not found. Please run feature_pipeline.py first.")
        st.stop()

    df = pd.read_parquet(FEATURE_STORE_PATH)
    df = normalize_columns(df)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    df = df.sort_values("time").reset_index(drop=True)

    return df, "Local Parquet Backup"


def load_model_bundle():
    if not MODEL_PATH.exists():
        st.error("Model not found. Please run training_pipeline.py first.")
        st.stop()

    bundle = joblib.load(MODEL_PATH)

    if not isinstance(bundle, dict) or "models" not in bundle:
        st.error("Invalid model bundle. Please run training_pipeline.py again.")
        st.stop()

    return bundle


def load_metrics():
    if not METRICS_PATH.exists():
        return None

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def create_city_column(df):
    df = df.copy()

    if "latitude" in df.columns and "longitude" in df.columns:
        df["location_lookup_key"] = (
            pd.to_numeric(df["latitude"], errors="coerce").round(4).astype(str)
            + ","
            + pd.to_numeric(df["longitude"], errors="coerce").round(4).astype(str)
        )
        mapped_city = df["location_lookup_key"].map(CITY_MAP)
    else:
        mapped_city = pd.Series([pd.NA] * len(df), index=df.index)

    if "city" in df.columns:
        df["city"] = df["city"].astype(str)
        df["city"] = df["city"].replace(
            ["None", "none", "nan", "NaN", "NULL", "null", "", "<NA>"],
            pd.NA,
        )
        df["city"] = df["city"].fillna(mapped_city)
    else:
        df["city"] = mapped_city

    df["city"] = df["city"].fillna(df.get("location_key", "Unknown"))
    df["city"] = df["city"].astype(str)
    df = df[df["city"].str.lower() != "none"].copy()

    return df


def get_model_info(model_bundle, target_key):
    models = model_bundle.get("models", {})

    if target_key not in models:
        st.error(f"Selected model not found: {target_key}")
        st.stop()

    model_info = models[target_key]
    model = model_info.get("model")

    if model is None:
        st.error(f"Model object not found for {target_key}. Run training_pipeline.py again.")
        st.stop()

    model_name = model_info.get("model_name", "unknown_model")
    feature_columns = model_info.get("feature_columns") or model_bundle.get("feature_columns")

    if not feature_columns:
        st.error("Model feature columns missing. Run training_pipeline.py again.")
        st.stop()

    return model, model_name, feature_columns


def prepare_city_dataframe_for_model(city_df, selected_city, feature_columns):
    prepared_df = city_df.copy()
    prepared_df = prepared_df.sort_values("time").reset_index(drop=True)

    for feature in feature_columns:
        if feature not in prepared_df.columns:
            if feature.startswith("city_"):
                prepared_df[feature] = 1 if feature == f"city_{selected_city}" else 0
            else:
                prepared_df[feature] = 0

    for feature in feature_columns:
        prepared_df[feature] = pd.to_numeric(prepared_df[feature], errors="coerce")
        median_value = prepared_df[feature].median()

        if pd.isna(median_value):
            median_value = 0

        prepared_df[feature] = prepared_df[feature].fillna(median_value)

    return prepared_df


def create_inference_frame(latest_row, feature_columns):
    values = []

    for feature in feature_columns:
        value = latest_row[feature] if feature in latest_row.index else 0
        if pd.isna(value):
            value = 0
        values.append(value)

    return pd.DataFrame([values], columns=feature_columns)


def predict_for_target(model_bundle, latest_row, target_key):
    model, model_name, feature_columns = get_model_info(model_bundle, target_key)
    X = create_inference_frame(latest_row, feature_columns)

    prediction = float(model.predict(X)[0])
    prediction = max(0, min(500, prediction))

    return prediction, model_name, feature_columns


def make_pollutant_table(latest_row, live_current=None):
    rows = []

    for col in POLLUTANT_COLUMNS:
        value = None

        if live_current and col in live_current and live_current[col] is not None:
            value = live_current[col]
        elif col in latest_row.index:
            value = latest_row[col]

        if value is not None and not pd.isna(value):
            rows.append(
                {
                    "Key": col,
                    "Name": POLLUTANT_NAMES.get(col, col),
                    "Short": POLLUTANT_SHORT.get(col, col),
                    "Value": round(float(value), 1),
                    "Unit": "µg/m³",
                    "Icon": POLLUTANT_ICONS.get(col, "🌫️"),
                    "Color": get_pollutant_color(col, float(value)),
                }
            )

    return pd.DataFrame(rows)


def get_selected_model_metrics(metrics_data, selected_target_key, selected_model_name=None):
    default_metrics = {
        "model_name": clean_model_name(selected_model_name) if selected_model_name else "Not Available",
        "model_key": selected_model_name or "not_available",
        "rmse": None,
        "mae": None,
        "r2": None,
    }

    if not isinstance(metrics_data, dict):
        return default_metrics

    target_metrics = metrics_data.get("targets", {}).get(selected_target_key, {})

    if not isinstance(target_metrics, dict):
        return default_metrics

    best_model_key = (
        target_metrics.get("best_model")
        or target_metrics.get("model_name")
        or selected_model_name
        or "not_available"
    )

    best_metrics = target_metrics.get("best_metrics") or target_metrics.get("metrics") or {}

    rmse = best_metrics.get("rmse")
    mae = best_metrics.get("mae")
    r2 = best_metrics.get("r2")

    all_models = target_metrics.get("all_models") or target_metrics.get("all_model_results") or {}

    if isinstance(all_models, dict):
        possible_keys = [
            best_model_key,
            str(best_model_key).lower(),
            str(best_model_key).replace(" ", "_").lower(),
        ]

        for key in possible_keys:
            if key in all_models and isinstance(all_models[key], dict):
                rmse = rmse if rmse is not None else all_models[key].get("rmse")
                mae = mae if mae is not None else all_models[key].get("mae")
                r2 = r2 if r2 is not None else all_models[key].get("r2")
                break

    return {
        "model_name": clean_model_name(best_model_key),
        "model_key": best_model_key,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


# =========================
# CHARTS
# =========================
def style_chart(fig, height=440):
    fig.update_layout(
        height=height,
        plot_bgcolor="#fffdf2",
        paper_bgcolor="#fffdf2",
        font=dict(color="#0f172a", family="Inter, Arial, sans-serif", size=13),
        title_font=dict(size=22, color="#0f172a", family="Inter, Arial, sans-serif"),
        margin=dict(l=25, r=25, t=65, b=45),
        xaxis=dict(gridcolor="#eadfba", zerolinecolor="#eadfba"),
        yaxis=dict(gridcolor="#eadfba", zerolinecolor="#eadfba"),
        hoverlabel=dict(bgcolor="#0f172a", font_color="white"),
    )
    return fig


def create_premium_trend_chart(chart_df, selected_city, current_aqi, predicted_aqi):
    chart_df = chart_df.copy()
    chart_df = chart_df.sort_values("time")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=chart_df["time"],
            y=chart_df["us_aqi"],
            mode="lines",
            name="Historical AQI",
            line=dict(color="#f59e0b", width=4, shape="spline", smoothing=0.65),
            fill="tozeroy",
            fillcolor="rgba(250, 204, 21, 0.18)",
            hovertemplate="<b>AQI</b>: %{y:.1f}<br><b>Time</b>: %{x}<extra></extra>",
        )
    )

    fig.add_hline(
        y=100,
        line_width=1,
        line_dash="dot",
        line_color="rgba(234, 179, 8, 0.55)",
    )

    fig.add_hline(
        y=150,
        line_width=1.5,
        line_dash="dash",
        line_color="rgba(239, 68, 68, 0.60)",
    )

    fig.add_trace(
        go.Scatter(
            x=[chart_df["time"].iloc[-1]],
            y=[current_aqi],
            mode="markers+text",
            name="Live AQI",
            marker=dict(size=17, color="#2563eb", line=dict(color="#ffffff", width=4)),
            text=[f"Live {round(current_aqi, 1)}"],
            textposition="top center",
            hovertemplate="<b>Live AQI</b>: %{y:.1f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[chart_df["time"].iloc[-1]],
            y=[predicted_aqi],
            mode="markers+text",
            name="Forecast",
            marker=dict(size=17, color="#ef4444", line=dict(color="#ffffff", width=4)),
            text=[f"Forecast {round(predicted_aqi, 1)}"],
            textposition="bottom center",
            hovertemplate="<b>Forecast AQI</b>: %{y:.1f}<extra></extra>",
        )
    )

    min_y = max(0, min(chart_df["us_aqi"].min(), current_aqi, predicted_aqi) - 18)
    max_y = min(500, max(chart_df["us_aqi"].max(), current_aqi, predicted_aqi) + 25)

    fig.update_layout(
        title=dict(
            text=f"Recent AQI Trend — {selected_city}",
            x=0.02,
            y=0.96,
            font=dict(size=24, color="#0f172a", family="Inter, Arial, sans-serif"),
        ),
        height=430,
        plot_bgcolor="#fffdf2",
        paper_bgcolor="#fffdf2",
        font=dict(color="#0f172a", family="Inter, Arial, sans-serif", size=13),
        margin=dict(l=35, r=35, t=80, b=50),
        xaxis=dict(
            title="Time",
            gridcolor="rgba(234, 223, 186, 0.85)",
            zerolinecolor="rgba(234, 223, 186, 0.85)",
        ),
        yaxis=dict(
            title="AQI-US",
            range=[min_y, max_y],
            gridcolor="rgba(234, 223, 186, 0.90)",
            zerolinecolor="rgba(234, 223, 186, 0.90)",
        ),
        hoverlabel=dict(bgcolor="#0f172a", font_color="white", bordercolor="#0f172a"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0)",
        ),
    )

    return fig


def create_city_comparison_chart(comparison_df):
    color_map = {
        "Good": "#22c55e",
        "Moderate": "#2563eb",
        "Unhealthy for Sensitive Groups": "#f97316",
        "Unhealthy": "#ef4444",
        "Very Unhealthy": "#8b5cf6",
        "Hazardous": "#111827",
    }

    fig = px.bar(
        comparison_df.sort_values("Live AQI", ascending=True),
        x="City",
        y="Live AQI",
        color="Category",
        color_discrete_map=color_map,
        text="Live AQI",
        title="Current AQI Comparison by City",
    )

    fig.update_traces(textposition="outside")
    return style_chart(fig, height=440)


# =========================
# RENDER HELPERS
# =========================
def render_forecast_card(title, prediction, category, icon, color, active=False):
    active_class = " active" if active else ""

    html = (
        f'<div class="metric-card{active_class}">'
        f'<div class="metric-title">{title}</div>'
        f'<div class="metric-value" style="color:{color};">{round(float(prediction), 1)}</div>'
        f'<div class="metric-desc">{icon} {category}</div>'
        f'</div>'
    )

    st.markdown(html, unsafe_allow_html=True)


def render_aqi_scale(current_aqi, predicted_aqi):
    live_percent = max(0, min(100, (float(current_aqi) / 500) * 100))
    forecast_percent = max(0, min(100, (float(predicted_aqi) / 500) * 100))

    html = (
        f'<div class="aqi-scale-card">'
        f'<div class="scale-track">'
        f'<div class="scale-marker-live" style="left:{live_percent}%;"></div>'
        f'<div class="scale-marker-forecast" style="left:{forecast_percent}%;"></div>'
        f'<div class="scale-badge" style="left:{forecast_percent}%;">Forecast {round(float(predicted_aqi), 1)}</div>'
        f'</div>'
        f'<div class="scale-numbers"><div>0</div><div>50</div><div>100</div><div>150</div><div>300</div><div>500</div></div>'
        f'<div class="scale-labels"><div>Good</div><div>Moderate</div><div>USG</div><div>Unhealthy</div><div>Very Unhealthy</div><div>Hazardous</div></div>'
        f'</div>'
    )

    st.markdown(html, unsafe_allow_html=True)


def render_pollutants(pollutant_df, selected_city):
    if pollutant_df.empty:
        st.info("Pollutant data is not available.")
        return

    cards_html = ""

    for _, row in pollutant_df.iterrows():
        cards_html += (
            f'<div class="pollutant-card" style="border-left-color:{row["Color"]};">'
            f'<div class="pollutant-left">'
            f'<div class="pollutant-icon">{row["Icon"]}</div>'
            f'<div>'
            f'<div class="pollutant-name">{row["Name"]}</div>'
            f'<div class="pollutant-short">({row["Short"]})</div>'
            f'</div>'
            f'</div>'
            f'<div class="pollutant-right">'
            f'<div class="pollutant-value">{row["Value"]}</div>'
            f'<div class="pollutant-unit">{row["Unit"]}</div>'
            f'</div>'
            f'</div>'
        )

    html = (
        f'<div class="premium-card">'
        f'<div class="premium-header">'
        f'<div>'
        f'<div class="premium-title">🌫️ Major Air Pollutants</div>'
        f'<div class="premium-subtitle">{selected_city} · live pollutant breakdown with a premium dashboard feel.</div>'
        f'</div>'
        f'<div class="premium-chip">Live breakdown</div>'
        f'</div>'
        f'<div class="pollutant-grid">{cards_html}</div>'
        f'</div>'
    )

    st.markdown(html, unsafe_allow_html=True)


def get_feature_importance(model_bundle, selected_target_key):
    try:
        model, _, feature_columns = get_model_info(model_bundle, selected_target_key)

        final_model = model

        if hasattr(model, "named_steps") and "model" in model.named_steps:
            final_model = model.named_steps["model"]

        if hasattr(final_model, "feature_importances_"):
            values = final_model.feature_importances_
        elif hasattr(final_model, "coef_"):
            values = np.abs(final_model.coef_)
        else:
            return pd.DataFrame()

        if hasattr(values, "ravel"):
            values = values.ravel()

        if len(values) != len(feature_columns):
            return pd.DataFrame()

        importance_df = pd.DataFrame(
            {
                "Feature": feature_columns,
                "Importance": values,
            }
        )

        importance_df = importance_df.sort_values("Importance", ascending=False)
        return importance_df

    except Exception:
        return pd.DataFrame()


def render_why_prediction(
    model_bundle,
    selected_target_key,
    selected_city,
    predicted_aqi,
    category,
    current_aqi,
    pm25_value,
    trend_status,
    future_weather_applied,
):
    html = (
        f'<div class="premium-card">'
        f'<div class="why-main">'
        f'<div class="premium-title">🧠 Why did the model predict this?</div>'
        f'<div class="premium-subtitle">'
        f'The forecast is mainly based on recent AQI, pollutant levels, pollution trend, and weather conditions. '
        f'For {selected_city}, the model predicts <b>{round(float(predicted_aqi), 1)}</b>, which falls under <b>{category}</b>.'
        f'</div>'
        f'</div>'
        f'<div class="why-grid">'
        f'<div class="why-card">'
        f'<div class="why-icon">📍</div>'
        f'<div class="why-title">Current air quality is important</div>'
        f'<div class="why-text">The latest AQI level is one of the strongest signals. When current AQI is already high, the model expects pollution risk to remain elevated.</div>'
        f'<div class="why-tag">Live AQI: {round(float(current_aqi), 1)}</div>'
        f'</div>'
        f'<div class="why-card">'
        f'<div class="why-icon">🌫️</div>'
        f'<div class="why-title">Fine particles are affecting the forecast</div>'
        f'<div class="why-text">PM2.5 particles are very small and can stay in the air longer. Higher PM2.5 usually increases health risk and forecasted AQI.</div>'
        f'<div class="why-tag">PM2.5: {pm25_value}</div>'
        f'</div>'
        f'<div class="why-card">'
        f'<div class="why-icon">📈</div>'
        f'<div class="why-title">Recent trend is guiding the model</div>'
        f'<div class="why-text">The model checks whether AQI has been rising, falling, or staying stable over recent hours. This helps it estimate the next day risk.</div>'
        f'<div class="why-tag">Trend: {trend_status}</div>'
        f'</div>'
        f'<div class="why-card">'
        f'<div class="why-icon">🌤️</div>'
        f'<div class="why-title">Weather can change pollution movement</div>'
        f'<div class="why-text">Wind, temperature, humidity, and pressure affect how pollution spreads or stays trapped near the city.</div>'
        f'<div class="why-tag">{"Live weather used" if future_weather_applied else "Saved weather used"}</div>'
        f'</div>'
        f'</div>'
        f'</div>'
    )

    st.markdown(html, unsafe_allow_html=True)

    importance_df = get_feature_importance(model_bundle, selected_target_key)

    if not importance_df.empty:
        with st.expander("Advanced: View technical model drivers"):
            fig = px.bar(
                importance_df.sort_values("Importance", ascending=True).tail(15),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top Model Drivers",
            )
            fig.update_traces(marker_color="#f59e0b")
            st.plotly_chart(style_chart(fig, height=520), use_container_width=True)
            st.dataframe(importance_df, use_container_width=True)


def build_city_comparison(df, model_bundle):
    rows = []

    for city in sorted(df["city"].dropna().unique()):
        city_df = df[df["city"] == city].copy()

        if city_df.empty:
            continue

        _, _, feature_columns = get_model_info(model_bundle, "day_1")
        city_df = prepare_city_dataframe_for_model(city_df, city, feature_columns)

        if city_df.empty:
            continue

        latest_row = city_df.sort_values("time").iloc[-1].copy()

        live_current = None

        if "latitude" in latest_row.index and "longitude" in latest_row.index:
            live_current = fetch_live_current_aqi(latest_row["latitude"], latest_row["longitude"])

        if live_current and live_current.get("us_aqi") is not None:
            current_aqi = float(live_current["us_aqi"])
        elif "us_aqi" in latest_row.index and not pd.isna(latest_row["us_aqi"]):
            current_aqi = float(latest_row["us_aqi"])
        else:
            continue

        try:
            predicted_aqi, _, _ = predict_for_target(model_bundle, latest_row, "day_1")
        except Exception:
            predicted_aqi = None

        category, _, icon, color = get_aqi_category(current_aqi)
        visual = get_city_visual(city)

        rows.append(
            {
                "City": city,
                "City Icon": visual["emoji"],
                "Landmark": visual["landmark"],
                "Live AQI": round(current_aqi, 1),
                "Predicted Day 1 AQI": round(predicted_aqi, 1) if predicted_aqi is not None else None,
                "Category": category,
                "Icon": icon,
                "Color": color,
            }
        )

    return pd.DataFrame(rows)


def render_city_ranking_cards(comparison_df):
    if comparison_df.empty:
        return

    display_df = comparison_df.sort_values("Live AQI", ascending=False).head(6).copy()
    cards_html = ""

    for i, (_, row) in enumerate(display_df.iterrows(), start=1):
        color = row.get("Color", "#64748b")
        cards_html += (
            f'<div class="rank-card">'
            f'<div class="rank-top">'
            f'<div class="rank-city">{row["City Icon"]} {row["City"]}</div>'
            f'<div class="rank-badge">Rank #{i}</div>'
            f'</div>'
            f'<div class="rank-aqi" style="color:{color};">{row["Live AQI"]}</div>'
            f'<div class="rank-status">{row["Icon"]} {row["Category"]}</div>'
            f'<div class="rank-landmark">{row["Landmark"]}</div>'
            f'</div>'
        )

    html = (
        '<div class="premium-card">'
        '<div class="premium-header">'
        '<div>'
        '<div class="premium-title">🏙️ UAE Metro Cities</div>'
        '<div class="premium-subtitle">Live city ranking with city-specific vibe and premium cards.</div>'
        '</div>'
        '<div class="premium-chip">City Ranking</div>'
        '</div>'
        f'<div class="ranking-grid">{cards_html}</div>'
        '</div>'
    )

    st.markdown(html, unsafe_allow_html=True)


def render_model_performance(metrics, selected_target_key, selected_day_label, selected_model_name):
    model_metrics = get_selected_model_metrics(
        metrics_data=metrics,
        selected_target_key=selected_target_key,
        selected_model_name=selected_model_name,
    )

    best_model = model_metrics.get("model_name", "N/A")
    best_rmse = model_metrics.get("rmse")
    best_mae = model_metrics.get("mae")
    best_r2 = model_metrics.get("r2")

    accuracy_label, accuracy_text, accuracy_color, accuracy_icon = get_accuracy_label(best_r2)

    rmse_text = "N/A" if best_rmse is None else f"± {round(float(best_rmse), 1)} AQI"
    mae_text = "N/A" if best_mae is None else f"{round(float(best_mae), 1)} AQI"
    r2_text = "N/A" if best_r2 is None else f"{round(float(best_r2) * 100, 1)}%"

    html = (
        '<div class="premium-card">'
        '<div class="premium-header">'
        '<div>'
        '<div class="premium-title">🤖 Model Performance</div>'
        '<div class="premium-subtitle">Simple summary of how reliable this forecast model is.</div>'
        '</div>'
        '<div class="premium-chip">ML Quality Check</div>'
        '</div>'
        '<div class="model-grid">'
        '<div class="model-card">'
        '<div class="model-icon">🤖</div>'
        '<div class="model-label">Best Model</div>'
        f'<div class="model-value">{best_model}</div>'
        '<div class="model-desc">This model performed best for the selected forecast.</div>'
        '</div>'
        '<div class="model-card">'
        f'<div class="model-icon" style="background:{accuracy_color}18;color:{accuracy_color};">{accuracy_icon}</div>'
        '<div class="model-label">Accuracy Level</div>'
        f'<div class="model-value" style="color:{accuracy_color};">{accuracy_label}</div>'
        f'<div class="model-desc">{accuracy_text}</div>'
        '</div>'
        '<div class="model-card">'
        '<div class="model-icon" style="background:#f9731618;color:#f97316;">🎯</div>'
        '<div class="model-label">Expected Error Range</div>'
        f'<div class="model-value">{rmse_text}</div>'
        '<div class="model-desc">Average uncertainty range around the forecast.</div>'
        '</div>'
        '<div class="model-card">'
        '<div class="model-icon" style="background:#0ea5e918;color:#0ea5e9;">📅</div>'
        '<div class="model-label">Forecast Target</div>'
        f'<div class="model-value">{clean_target_name(selected_target_key)}</div>'
        f'<div class="model-desc">Currently selected: {selected_day_label}</div>'
        '</div>'
        '</div>'
        '<div class="model-explain-row">'
        '<div class="model-explain-box"><b>RMSE</b><br>'
        f'The model prediction is usually around <b>{rmse_text}</b> away from the actual AQI.</div>'
        '<div class="model-explain-box"><b>MAE</b><br>'
        f'On average, the model misses the real AQI by about <b>{mae_text}</b>.</div>'
        '<div class="model-explain-box"><b>R² Score</b><br>'
        f'The model explains about <b>{r2_text}</b> of the AQI pattern for this forecast.</div>'
        '</div>'
        '</div>'
    )

    st.markdown(html, unsafe_allow_html=True)


def render_technical_summary(
    feature_store_source,
    selected_day_label,
    selected_model_name,
    location_df,
    current_source,
    future_weather_applied,
    latest_time,
    model_metrics,
    model_feature_columns,
):
    html = (
        '<div class="premium-card">'
        '<div class="premium-header">'
        '<div>'
        '<div class="premium-title">🛠️ Technical Summary</div>'
        '<div class="premium-subtitle">Clean project-level metadata for validation, report writing, and debugging.</div>'
        '</div>'
        '<div class="premium-chip">System Details</div>'
        '</div>'
        '<div class="tech-grid">'
        f'<div class="tech-box"><div class="tech-label">Feature Store</div><div class="tech-value">{feature_store_source}</div></div>'
        f'<div class="tech-box"><div class="tech-label">Selected Forecast</div><div class="tech-value">{selected_day_label}</div></div>'
        f'<div class="tech-box"><div class="tech-label">Selected Model</div><div class="tech-value">{selected_model_name}</div></div>'
        f'<div class="tech-box"><div class="tech-label">Rows for City</div><div class="tech-value">{len(location_df)}</div></div>'
        f'<div class="tech-box"><div class="tech-label">Live AQI Source</div><div class="tech-value">{current_source}</div></div>'
        f'<div class="tech-box"><div class="tech-label">Future Weather Applied</div><div class="tech-value">{future_weather_applied}</div></div>'
        f'<div class="tech-box"><div class="tech-label">Model Feature Timestamp</div><div class="tech-value">{str(latest_time)[:16]}</div></div>'
        f'<div class="tech-box"><div class="tech-label">RMSE / MAE / R²</div><div class="tech-value">{round(float(model_metrics.get("rmse")), 2) if model_metrics.get("rmse") is not None else "N/A"} / {round(float(model_metrics.get("mae")), 2) if model_metrics.get("mae") is not None else "N/A"} / {round(float(model_metrics.get("r2")), 3) if model_metrics.get("r2") is not None else "N/A"}</div></div>'
        f'<div class="tech-box"><div class="tech-label">Feature Count</div><div class="tech-value">{len(model_feature_columns)}</div></div>'
        '</div>'
        '</div>'
    )

    st.markdown(html, unsafe_allow_html=True)


def render_premium_footer(selected_city, latest_time, feature_store_source):
    city_visual = get_city_visual(selected_city)

    html = (
        '<div class="footer-premium">'
        '<div class="footer-grid">'
        '<div class="footer-box">'
        '<div class="footer-main-title">🌍 Premium AQI Dashboard</div>'
        '<div class="footer-main-subtitle">'
        f'Live + predictive AQI intelligence for <b>{city_visual["emoji"]} {selected_city}</b>. '
        'Designed with a premium yellow-blue theme for clear forecasting, health awareness, and pollutant monitoring.'
        '</div>'
        '</div>'
        '<div class="footer-box">'
        '<div class="footer-title">✨ Highlights</div>'
        '<div class="footer-item">• Real-time AQI monitoring</div>'
        '<div class="footer-item">• Day 1 / Day 2 / Day 3 forecasts</div>'
        '<div class="footer-item">• Health recommendations & pollutant cards</div>'
        '<div class="footer-item">• City comparison with UAE vibe icons</div>'
        '</div>'
        '<div class="footer-box">'
        '<div class="footer-title">⚙️ Stack</div>'
        '<div class="footer-item">• Python + Streamlit</div>'
        '<div class="footer-item">• Plotly + Scikit-learn</div>'
        '<div class="footer-item">• Open-Meteo APIs</div>'
        f'<div class="footer-item">• Source: {feature_store_source}</div>'
        '</div>'
        '</div>'
        '<div class="footer-bottom">'
        f'<div>📍 Active City: {city_visual["emoji"]} {selected_city} · {city_visual["landmark"]}</div>'
        f'<div>🕒 Historical feature snapshot: {str(latest_time)[:16]}</div>'
        '<div>Made with 💛 for a premium AQI forecasting experience</div>'
        '</div>'
        '</div>'
    )

    st.markdown(html, unsafe_allow_html=True)


# =========================
# MAIN APP
# =========================
def main():
    setup_page()

    df, feature_store_source = load_feature_store()
    print(f"Feature store source used: {feature_store_source}")
    print(f"Feature store rows loaded: {len(df) if df is not None else 0}")
    df = create_city_column(df)

    model_bundle = load_model_bundle()
    metrics = load_metrics()

    available_cities = sorted(df["city"].dropna().unique())

    if not available_cities:
        st.error("No cities found in feature store.")
        st.stop()

    default_city = "Dubai" if "Dubai" in available_cities else available_cities[0]

    st.sidebar.title("Dashboard Controls")

    selected_city = st.sidebar.radio(
        "Select City",
        available_cities,
        index=available_cities.index(default_city),
        format_func=lambda x: f'{get_city_visual(x)["emoji"]} {x}',
    )

    selected_day_label = st.sidebar.radio(
        "Select Forecast Day",
        list(DAY_OPTIONS.keys()),
        index=0,
        format_func=lambda x: f'{DAY_ICONS.get(x, "📌")} {x}',
    )

    selected_target_key = DAY_OPTIONS[selected_day_label]

    _, selected_model_name, model_feature_columns = get_model_info(
        model_bundle,
        selected_target_key,
    )

    location_df = df[df["city"] == selected_city].copy()

    if location_df.empty:
        st.error(f"No data found for {selected_city}.")
        st.stop()

    location_df = prepare_city_dataframe_for_model(
        location_df,
        selected_city,
        model_feature_columns,
    )

    latest_row = location_df.iloc[-1].copy()
    latest_time = latest_row["time"]

    latest_latitude = float(latest_row["latitude"]) if "latitude" in latest_row.index else 25.2048
    latest_longitude = float(latest_row["longitude"]) if "longitude" in latest_row.index else 55.2708

    future_weather_df = fetch_future_weather(latest_latitude, latest_longitude)
    latest_row, future_weather_applied = update_row_with_future_weather(latest_row, future_weather_df)

    live_current = fetch_live_current_aqi(latest_latitude, latest_longitude)
    live_weather = fetch_live_current_weather(latest_latitude, latest_longitude)

    if live_current and live_current.get("us_aqi") is not None:
        current_aqi = float(live_current["us_aqi"])
        current_source = "Live Open-Meteo current AQI"
        live_time = live_current.get("time")
    else:
        current_aqi = float(latest_row["us_aqi"]) if "us_aqi" in latest_row.index else 0
        current_source = "Saved feature store AQI"
        live_time = None

    current_category, current_message, current_icon, current_color = get_aqi_category(current_aqi)

    day_predictions = {}

    for target_key in DAY_OPTIONS.values():
        prediction, model_name, _ = predict_for_target(model_bundle, latest_row, target_key)
        category, message, icon, color = get_aqi_category(prediction)

        day_predictions[target_key] = {
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
    color = selected_prediction["color"]

    model_metrics = get_selected_model_metrics(
        metrics,
        selected_target_key,
        selected_model_name,
    )

    selected_rmse = model_metrics.get("rmse")

    if selected_rmse is not None:
        lower_bound = max(0, predicted_aqi - float(selected_rmse))
        upper_bound = min(500, predicted_aqi + float(selected_rmse))
    else:
        lower_bound = max(0, predicted_aqi - 15)
        upper_bound = min(500, predicted_aqi + 15)

    trend_change = round(predicted_aqi - current_aqi, 1)

    if trend_change > 2:
        trend_status = "Worsening"
        trend_full = f"▲ Worsening (+{abs(trend_change)})"
        trend_color = "#ef4444"
    elif trend_change < -2:
        trend_status = "Improving"
        trend_full = f"▼ Improving (-{abs(trend_change)})"
        trend_color = "#22c55e"
    else:
        trend_status = "Stable"
        trend_full = "→ Stable"
        trend_color = "#2563eb"

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

    city_visual = get_city_visual(selected_city)

    # -------------------------
    # TOP BRAND ROW
    # -------------------------
    st.markdown(
        '<div class="brand-row">'
        '<div class="brand-left">'
        '<div class="brand-logo">AQ</div>'
        '<div class="brand-title">AQI Forecast</div>'
        '</div>'
        '<div class="brand-pills">'
        '<div class="brand-pill">AQI-US Standard</div>'
        '<div class="brand-pill">UAE Cities</div>'
        '<div class="brand-pill">ML Forecast</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # -------------------------
    # HERO
    # -------------------------
    st.markdown(
        f'<div class="hero-yellow">'
        f'<div class="live-pill">● LIVE</div>'
        f'<div class="city-vibe">{city_visual["emoji"]} {selected_city} · {city_visual["landmark"]}</div>'
        f'<div class="hero-title">{selected_city} Air Quality Index Forecast</div>'
        f'<div class="hero-subtitle">Real-time AQI monitoring with machine learning based {selected_day_label.lower()} prediction, pollutant analytics, city comparison, and premium dashboard experience.</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # -------------------------
    # TOP CARDS
    # -------------------------
    col1, col2, col3 = st.columns([1.05, 1.05, 1])

    with col1:
        st.markdown(
            f'<div class="top-card">'
            f'<div class="top-card-label">🌫️ Live AQI</div>'
            f'<div class="live-number">{round(current_aqi, 1)}</div>'
            f'<div class="top-card-desc">AQI-US Standard · {current_icon} {current_category}</div>'
            f'<div class="mini-row"><span>🧪 PM2.5: {pm25_value}</span><span>💨 PM10: {pm10_value}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f'<div class="top-card">'
            f'<div class="top-card-label">🎯 Forecast AQI</div>'
            f'<div class="forecast-number-label">Predicted AQI</div>'
            f'<div class="forecast-number">{round(predicted_aqi, 1)}</div>'
            f'<div class="forecast-category-badge"><span class="status-dot"></span><span>{category}</span></div>'
            f'<div class="forecast-details">'
            f'<b>Expected range:</b> {round(lower_bound, 1)} – {round(upper_bound, 1)}<br>'
            f'<b>Direction:</b> <span style="color:{trend_color};font-weight:900;">{trend_full}</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col3:
        temp_value = live_weather.get("temperature_2m") if live_weather else latest_row.get("temperature_2m", None)
        humidity_value = live_weather.get("relative_humidity_2m") if live_weather else latest_row.get("relative_humidity_2m", None)
        wind_value = live_weather.get("wind_speed_10m") if live_weather else latest_row.get("wind_speed_10m", None)

        temp_display = f"{round(float(temp_value), 1)}°C" if temp_value is not None and not pd.isna(temp_value) else "N/A"
        humidity_display = f"{round(float(humidity_value), 1)}%" if humidity_value is not None and not pd.isna(humidity_value) else "N/A"
        wind_display = f"{round(float(wind_value), 1)}" if wind_value is not None and not pd.isna(wind_value) else "N/A"

        st.markdown(
            f'<div class="top-card">'
            f'<div class="weather-icon">🌤️</div>'
            f'<div class="weather-temp">{temp_display}</div>'
            f'<div class="top-card-desc"><b>Current / forecast weather inputs</b></div>'
            f'<div class="weather-grid">'
            f'<div class="weather-chip"><div class="weather-chip-label">💧 Humidity</div><div class="weather-chip-value">{humidity_display}</div></div>'
            f'<div class="weather-chip"><div class="weather-chip-label">🍃 Wind</div><div class="weather-chip-value">{wind_display}</div></div>'
            f'<div class="weather-chip"><div class="weather-chip-label">📡 Source</div><div class="weather-chip-value">Live</div></div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # -------------------------
    # INFO STRIP
    # -------------------------
    feature_store_source_text = str(feature_store_source or "").strip()

    if feature_store_source_text == "Hopsworks Feature Store":
        feature_store_source_display = "Hopsworks Feature Store"
    else:
        feature_store_source_display = "Local Parquet Backup"

    hopsworks_feature_group_name = str(get_setting("FEATURE_GROUP_NAME", "aqi_features_clean")).strip()
    hopsworks_feature_group_version = str(get_setting("FEATURE_GROUP_VERSION", "2")).strip()

    if not hopsworks_feature_group_name:
        hopsworks_feature_group_name = "aqi_features_clean"

    if not hopsworks_feature_group_version:
        hopsworks_feature_group_version = "2"

    st.markdown(
        f'<div class="source-strip">'
        f'<b>📡 Live AQI source:</b> {current_source} · <b>🕒 Live AQI timestamp:</b> {str(live_time) if live_time else "N/A"}<br>'
        f'<b>🌤️ Future weather source:</b> {"Live Open-Meteo Forecast API" if future_weather_applied else "Saved feature store values"}<br>'
        f'<b>🧾 Historical model feature timestamp:</b> {str(latest_time)[:16]}'
        f'<br><b>Feature Store:</b> {feature_store_source_display}'
        f'<br><b>Feature Group:</b> {hopsworks_feature_group_name} v{hopsworks_feature_group_version}'
        f'<br><b>Model Artifact:</b> Hopsworks Resources / aqi_model_artifacts'
        f'<br><b>Model File:</b> aqi_model.joblib'
        f'<br><b>Metrics File:</b> metrics.json'
        f'</div>',
        unsafe_allow_html=True,
    )

    try:
        latest_time_for_check = pd.to_datetime(latest_time, errors="coerce")

        if pd.notna(latest_time_for_check):
            if getattr(latest_time_for_check, "tzinfo", None) is not None:
                current_time_for_check = pd.Timestamp.now(tz=latest_time_for_check.tzinfo)
            else:
                current_time_for_check = pd.Timestamp.now()

            if current_time_for_check - latest_time_for_check > pd.Timedelta(days=3):
                st.warning(
                    "Historical feature-store row is older than 3 days. Live AQI/weather is still being used for current display, but run feature_pipeline.py for fresher historical features."
                )

    except Exception as error:
        print(f"Timestamp freshness check skipped: {error}")

    st.info(message)

    # -------------------------
    # FORECAST SUMMARY
    # -------------------------
    st.markdown('<div class="section-title">Forecast Summary</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="aqiin-mini-grid">'
        f'<div class="aqiin-mini-card"><div class="aqiin-mini-label">🌫️ Current AQI</div><div class="aqiin-mini-value">{round(current_aqi, 1)}</div><div class="aqiin-mini-status">{current_category}</div></div>'
        f'<div class="aqiin-mini-card"><div class="aqiin-mini-label">📈 {selected_day_label}</div><div class="aqiin-mini-value">{round(predicted_aqi, 1)}</div><div class="aqiin-mini-status" style="color:{trend_color};">{trend_full}</div></div>'
        f'<div class="aqiin-mini-card"><div class="aqiin-mini-label">🚨 Risk Category</div><div class="aqiin-mini-value" style="font-size:25px;color:{color};">{category}</div><div class="aqiin-mini-status">Predicted risk level</div></div>'
        f'<div class="aqiin-mini-card"><div class="aqiin-mini-label">📊 Prediction Range</div><div class="aqiin-mini-value" style="font-size:29px;">{round(lower_bound, 1)} – {round(upper_bound, 1)}</div><div class="aqiin-mini-status">Model uncertainty</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # -------------------------
    # AQI SCALE
    # -------------------------
    st.markdown('<div class="section-title">AQI Scale</div>', unsafe_allow_html=True)
    render_aqi_scale(current_aqi, predicted_aqi)

    # -------------------------
    # 3-DAY OVERVIEW
    # -------------------------
    st.markdown('<div class="section-title">3-Day Forecast Overview</div>', unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4)

    with f1:
        item = day_predictions["day_1"]
        render_forecast_card("📍 Day 1 Forecast", item["prediction"], item["category"], item["icon"], item["color"], selected_target_key == "day_1")

    with f2:
        item = day_predictions["day_2"]
        render_forecast_card("📅 Day 2 Forecast", item["prediction"], item["category"], item["icon"], item["color"], selected_target_key == "day_2")

    with f3:
        item = day_predictions["day_3"]
        render_forecast_card("🧭 Day 3 Forecast", item["prediction"], item["category"], item["icon"], item["color"], selected_target_key == "day_3")

    with f4:
        item = day_predictions["next_72h_avg"]
        render_forecast_card("📊 3-Day Average", item["prediction"], item["category"], item["icon"], item["color"], selected_target_key == "next_72h_avg")

    # -------------------------
    # HEALTH RECOMMENDATIONS
    # -------------------------
    st.markdown('<div class="section-title">Health Recommendations</div>', unsafe_allow_html=True)

    recommendations = get_health_recommendations(predicted_aqi)
    r1, r2, r3 = st.columns(3)

    for index, recommendation in enumerate(recommendations):
        with [r1, r2, r3][index % 3]:
            st.markdown(
                f'<div class="recommend-card"><h4>Recommendation {index + 1}</h4><p>{recommendation}</p></div>',
                unsafe_allow_html=True,
            )

    # -------------------------
    # AQI TREND
    # -------------------------
    st.markdown('<div class="section-title">AQI Trend</div>', unsafe_allow_html=True)

    trend_df = location_df[["time", "us_aqi"]].dropna().tail(240).copy()

    if not trend_df.empty:
        st.markdown(
            f'<div class="trend-intro-card">'
            f'<div class="premium-header">'
            f'<div>'
            f'<div class="premium-title">📈 AQI Trend Intelligence</div>'
            f'<div class="premium-subtitle">Premium historical AQI movement with live AQI and selected forecast risk overlay.</div>'
            f'</div>'
            f'<div class="premium-chip">Trend Intelligence</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="trend-chart-shell">', unsafe_allow_html=True)
        st.plotly_chart(
            create_premium_trend_chart(trend_df, selected_city, current_aqi, predicted_aqi),
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            f'<div class="trend-stat-grid">'
            f'<div class="trend-stat"><div class="trend-stat-label">🔵 Live AQI</div><div class="trend-stat-value">{round(current_aqi, 1)}</div></div>'
            f'<div class="trend-stat"><div class="trend-stat-label">🔴 Forecast AQI</div><div class="trend-stat-value">{round(predicted_aqi, 1)}</div></div>'
            f'<div class="trend-stat"><div class="trend-stat-label">⚡ Direction</div><div class="trend-stat-value" style="color:{trend_color};">{trend_full}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("AQI trend data is not available.")

    # -------------------------
    # POLLUTANTS
    # -------------------------
    render_pollutants(pollutant_df, selected_city)

    # -------------------------
    # CITY COMPARISON
    # -------------------------
    st.markdown('<div class="section-title">City Comparison</div>', unsafe_allow_html=True)

    comparison_df = build_city_comparison(df, model_bundle)

    if not comparison_df.empty:
        render_city_ranking_cards(comparison_df)
        st.plotly_chart(create_city_comparison_chart(comparison_df), use_container_width=True)

        best_city = comparison_df.sort_values("Live AQI", ascending=True).iloc[0]
        worst_city = comparison_df.sort_values("Live AQI", ascending=False).iloc[0]

        c1, c2 = st.columns(2)

        with c1:
            st.success(
                f'Best current air quality: {best_city["City Icon"]} {best_city["City"]} with AQI {best_city["Live AQI"]}'
            )

        with c2:
            st.error(
                f'Highest current AQI: {worst_city["City Icon"]} {worst_city["City"]} with AQI {worst_city["Live AQI"]}'
            )

        with st.expander("View city comparison table"):
            st.dataframe(comparison_df, use_container_width=True)

    # -------------------------
    # WHY THIS PREDICTION
    # -------------------------
    render_why_prediction(
        model_bundle=model_bundle,
        selected_target_key=selected_target_key,
        selected_city=selected_city,
        predicted_aqi=predicted_aqi,
        category=category,
        current_aqi=current_aqi,
        pm25_value=pm25_value,
        trend_status=trend_full,
        future_weather_applied=future_weather_applied,
    )

    # -------------------------
    # MODEL PERFORMANCE
    # -------------------------
    render_model_performance(
        metrics,
        selected_target_key,
        selected_day_label,
        selected_model_name,
    )

    # -------------------------
    # DOWNLOADS
    # -------------------------
    st.markdown('<div class="section-title">Downloads</div>', unsafe_allow_html=True)

    latest_prediction_df = pd.DataFrame(
        [
            {
                "city": selected_city,
                "forecast": selected_day_label,
                "feature_store_source": feature_store_source,
                "model_source": "Local model backup",
                "live_current_aqi": current_aqi,
                "live_current_source": current_source,
                "future_weather_applied": future_weather_applied,
                "historical_feature_timestamp": str(latest_time),
                "predicted_aqi": predicted_aqi,
                "prediction_lower_bound": lower_bound,
                "prediction_upper_bound": upper_bound,
                "direction": trend_status,
                "category": category,
                "model": selected_model_name,
                "rmse": model_metrics.get("rmse"),
                "mae": model_metrics.get("mae"),
                "r2": model_metrics.get("r2"),
            }
        ]
    )

    d1, d2 = st.columns(2)

    with d1:
        st.download_button(
            label="Download Latest Prediction CSV",
            data=latest_prediction_df.to_csv(index=False),
            file_name=f"{selected_city.lower().replace(' ', '_')}_{selected_target_key}_aqi_prediction.csv",
            mime="text/csv",
        )

    with d2:
        if metrics:
            st.download_button(
                label="Download Model Metrics JSON",
                data=json.dumps(metrics, indent=4),
                file_name="model_metrics.json",
                mime="application/json",
            )

    # -------------------------
    # TECHNICAL SUMMARY
    # -------------------------
    render_technical_summary(
        feature_store_source=feature_store_source,
        selected_day_label=selected_day_label,
        selected_model_name=selected_model_name,
        location_df=location_df,
        current_source=current_source,
        future_weather_applied=future_weather_applied,
        latest_time=latest_time,
        model_metrics=model_metrics,
        model_feature_columns=model_feature_columns,
    )

    with st.expander("Advanced: Latest Model Feature Values"):
        available_features = [feature for feature in model_feature_columns if feature in latest_row.index]
        if available_features:
            latest_feature_df = latest_row[available_features].to_frame(name="Latest Value")
            st.dataframe(latest_feature_df, use_container_width=True)
        else:
            st.warning("No model feature values available.")

    with st.expander("Advanced: Recent Clean Feature Snapshot"):
        clean_columns = [
            "time",
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
            "wind_speed_10m",
            "surface_pressure",
        ]
        clean_columns = [column for column in clean_columns if column in location_df.columns]
        st.dataframe(location_df[clean_columns].tail(20), use_container_width=True)

    # -------------------------
    # PREMIUM FOOTER
    # -------------------------
    render_premium_footer(
        selected_city=selected_city,
        latest_time=latest_time,
        feature_store_source=feature_store_source,
    )


if __name__ == "__main__":
    main()
