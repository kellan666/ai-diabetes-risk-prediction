import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load only the trained model
model = joblib.load("diabetes_model.pkl")

st.set_page_config(
    page_title="AI Diabetes Risk Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 AI Diabetes Risk Prediction")
st.write("Enter the health information below to estimate diabetes risk.")

st.divider()

pregnancies = st.number_input(
    "Pregnancies",
    min_value=0,
    max_value=20,
    value=1
)

glucose = st.number_input(
    "Glucose Level (mg/dL)",
    min_value=0,
    max_value=300,
    value=120
)

blood_pressure = st.number_input(
    "Blood Pressure (mm Hg)",
    min_value=0,
    max_value=200,
    value=70
)

skin_thickness = st.number_input(
    "Skin Thickness (mm)",
    min_value=0,
    max_value=100,
    value=20
)

insulin = st.number_input(
    "Insulin (μU/mL)",
    min_value=0,
    max_value=900,
    value=80
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    max_value=70.0,
    value=25.0
)

diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.5
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=25
)

st.divider()

if st.button("🔍 Predict Diabetes Risk", use_container_width=True):

    # Create input DataFrame
    input_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age
    }])

    # IMPORTANT:
    # These values reproduce the preprocessing used during model training.
    medians = {
        "Glucose": 117.0,
        "BloodPressure": 72.0,
        "SkinThickness": 23.0,
        "Insulin": 125.0,
        "BMI": 32.0
    }

    for column, median_value in medians.items():
        if input_data[column].iloc[0] == 0:
            input_data[column] = median_value

    # Scaling parameters from the training dataset
    means = np.array([
        3.845,
        120.895,
        69.105,
        20.536,
        79.799,
        31.993,
        0.472,
        33.241
    ])

    stds = np.array([
        3.367,
        31.973,
        19.356,
        15.952,
        115.244,
        7.878,
        0.331,
        11.760
    ])

    X = input_data.values.astype(float)

    # Standardize exactly like StandardScaler
    X_scaled = (X - means) / stds

    # Prediction
    prediction = model.predict(X_scaled)
    probability = model.predict_proba(X_scaled)[0][1] * 100

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ Higher Diabetes Risk")
    else:
        st.success("✅ Lower Diabetes Risk")

    st.metric(
        "Estimated Risk Probability",
        f"{probability:.2f}%"
    )

    st.info(
        "This AI prediction is for educational and project demonstration "
        "purposes only and should not be used as a medical diagnosis."
    )
