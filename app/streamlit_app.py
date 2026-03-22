import streamlit as st
import joblib
import json
import numpy as np
import os

# Load model and encoders
base_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model     = joblib.load(os.path.join(base_dir, 'models/best_model.pkl'))
le_driver = joblib.load(os.path.join(base_dir, 'models/le_Driver.pkl'))
le_team   = joblib.load(os.path.join(base_dir, 'models/le_Team.pkl'))
le_gp     = joblib.load(os.path.join(base_dir, 'models/le_GrandPrix.pkl'))

st.set_page_config(page_title="F1 Quali Predictor 🏎️", layout="centered")
st.title("🏎️ F1 Qualifying Lap Time Predictor")
st.markdown("Enter practice session data to predict a driver's qualifying lap time.")

col1, col2 = st.columns(2)
with col1:
    driver = st.selectbox("Driver",  le_driver.classes_)
    team   = st.selectbox("Team",    le_team.classes_)
    gp     = st.selectbox("Circuit", le_gp.classes_)
with col2:
    fp1_best = st.number_input("FP1 Best Lap (s)", value=90.0)
    fp2_best = st.number_input("FP2 Best Lap (s)", value=89.5)
    fp3_best = st.number_input("FP3 Best Lap (s)", value=89.0)

st.subheader("Weather Conditions")
air_temp   = st.slider("Air Temp (°C)",   15, 40, 25)
track_temp = st.slider("Track Temp (°C)", 20, 60, 35)
humidity   = st.slider("Humidity (%)",    10, 100, 50)
rainfall   = st.checkbox("Rainfall?")

if st.button("🔮 Predict Qualifying Time"):
    features = np.array([[
        fp1_best, fp2_best, fp3_best,
        fp1_best, fp2_best, fp3_best,
        0.3, 0.3, 0.3,
        fp1_best - fp3_best,
        air_temp, track_temp, humidity, int(rainfall),
        le_driver.transform([driver])[0],
        le_team.transform([team])[0],
        le_gp.transform([gp])[0]
    ]])
    prediction = model.predict(features)[0]
    mins = int(prediction // 60)
    secs = prediction % 60
    st.success(f"⏱️ Predicted Qualifying Lap Time: **{prediction:.3f} seconds**")
    st.info(f"That's **{mins}:{secs:06.3f}** in F1 format")