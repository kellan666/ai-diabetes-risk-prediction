
import streamlit as st
import pandas as pd
import joblib

# Load trained model and preprocessing tools
model = joblib.load("diabetes_model.pkl")
imputer = joblib.load("diabetes_imputer.pkl")
scaler = joblib.load("diabetes_scaler.pkl")

# Page configuration
st.set_page_config(
    page_title="AI Diabetes Risk Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 AI Diabetes Risk Prediction")
st.write("Enter the health information below to estimate diabetes risk.")

st.divider()

# Input fields
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

# Prediction button
if st.button("🔍 Predict Diabetes Risk", use_container_width=True):

    data = [[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]]

    # Convert to DataFrame
    input_data = pd.DataFrame(
        data,
        columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]
    )

    # Apply preprocessing
    input_imputed = imputer.transform(input_data)
    input_scaled = scaler.transform(input_imputed)

    # Prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1] * 100

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
