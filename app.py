import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")
encoders = joblib.load("label_encoders.pkl")

st.title("🌙 Early Waker Prediction System")

# inputs
age = st.number_input("Age", 0.0)
gender = st.selectbox("Gender", ["Male", "Female"])
sleep = st.number_input("Sleep Duration", 0.0)
sleep_quality = st.number_input("Sleep Quality Score", 0.0)
stress = st.number_input("Stress Level", 0.0)
steps = st.number_input("Daily Steps", 0.0)
exercise = st.number_input("Exercise Frequency", 0.0)
health = st.number_input("Health Score", 0.0)
bmi = st.number_input("BMI", 0.0)

if st.button("Predict"):

    input_dict = {col: 0 for col in features}

    input_dict["Age"] = age
    input_dict["Sleep_Duration_Hours"] = sleep
    input_dict["Sleep_Quality_Score"] = sleep_quality
    input_dict["Stress_Level"] = stress
    input_dict["Daily_Steps"] = steps
    input_dict["Exercise_Frequency_Per_Week"] = exercise
    input_dict["Health_Score"] = health
    input_dict["BMI"] = bmi

    input_dict["Gender"] = encoders["Gender"].transform([gender])[0]

    input_df = pd.DataFrame([input_dict])[features]
    input_scaled = scaler.transform(input_df)

    prob = model.predict_proba(input_scaled)[0][1]
    ml_result = "Yes" if prob > 0.55 else "No"

    # RULE SYSTEM
    risk = 0
    if sleep < 5: risk += 2
    if sleep_quality < 40: risk += 2
    if stress > 7: risk += 2
    if steps < 3000: risk += 1
    if exercise < 2: risk += 1
    if health < 40: risk += 2
    if bmi > 30: risk += 2

    rule_result = "No" if risk >= 5 else "Yes"

    # FINAL DECISION
    final_result = rule_result

    st.success(f"🌟 Prediction: {final_result}")
    st.write(f"📊 Probability: {prob:.2f}")