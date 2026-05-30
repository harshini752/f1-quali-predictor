from pathlib import Path

import joblib
import numpy as np
import streamlit as st
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent.resolve()
MODELS_DIR = BASE_DIR / "models"
CONFIG_PATH = BASE_DIR / "config.yaml"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
with open(CONFIG_PATH) as fh:
    cfg = yaml.safe_load(fh)

_app = cfg["app"]
_sliders = _app["sliders"]
_defaults = _app["defaults"]

# ---------------------------------------------------------------------------
# Models (loaded once at startup)
# ---------------------------------------------------------------------------
encoder_prefix = cfg["paths"]["encoder_prefix"]
model = joblib.load(MODELS_DIR / "best_model.pkl")
le_driver = joblib.load(MODELS_DIR / f"{encoder_prefix}driver.pkl")
le_team = joblib.load(MODELS_DIR / f"{encoder_prefix}team.pkl")
le_gp = joblib.load(MODELS_DIR / f"{encoder_prefix}grandprix.pkl")

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title=_app["page_title"] + " 🏎️", layout="centered")
st.title("🏎️ F1 Qualifying Lap Time Predictor")
st.markdown("Enter practice session data to predict a driver's qualifying lap time.")

col1, col2 = st.columns(2)
with col1:
    driver = st.selectbox("Driver", le_driver.classes_)
    team = st.selectbox("Team", le_team.classes_)
    gp = st.selectbox("Circuit", le_gp.classes_)
with col2:
    fp1_best = st.number_input("FP1 Best Lap (s)", value=_defaults["fp1_best"])
    fp2_best = st.number_input("FP2 Best Lap (s)", value=_defaults["fp2_best"])
    fp3_best = st.number_input("FP3 Best Lap (s)", value=_defaults["fp3_best"])

st.subheader("Weather Conditions")
air_temp = st.slider(
    "Air Temp (°C)",
    _sliders["air_temp"]["min"],
    _sliders["air_temp"]["max"],
    _sliders["air_temp"]["default"],
)
track_temp = st.slider(
    "Track Temp (°C)",
    _sliders["track_temp"]["min"],
    _sliders["track_temp"]["max"],
    _sliders["track_temp"]["default"],
)
humidity = st.slider(
    "Humidity (%)",
    _sliders["humidity"]["min"],
    _sliders["humidity"]["max"],
    _sliders["humidity"]["default"],
)
rainfall = st.checkbox("Rainfall?")

if st.button("🔮 Predict Qualifying Time"):
    features = np.array([[
        fp1_best, fp2_best, fp3_best,
        fp1_best, fp2_best, fp3_best,
        0.3, 0.3, 0.3,
        fp1_best - fp3_best,
        air_temp, track_temp, humidity, int(rainfall),
        le_driver.transform([driver])[0],
        le_team.transform([team])[0],
        le_gp.transform([gp])[0],
    ]])
    prediction = model.predict(features)[0]
    mins = int(prediction // 60)
    secs = prediction % 60
    st.success(f"⏱️ Predicted Qualifying Lap Time: **{prediction:.3f} seconds**")
    st.info(f"That's **{mins}:{secs:06.3f}** in F1 format")
