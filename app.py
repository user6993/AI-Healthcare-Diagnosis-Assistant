import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("model/disease_model.pkl")
target_encoder = joblib.load("model/target_encoder.pkl")

st.set_page_config(
    page_title="AI Healthcare Diagnosis Assistant",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 AI Healthcare Diagnosis Assistant")

st.write("Enter Patient Details")

age = st.number_input("Age", 1, 120, 30)

gender = st.selectbox(
    "Gender",
    ["Female", "Male", "Other"]
)

bp = st.number_input("Blood Pressure", 80, 220, 120)

heart = st.number_input("Heart Rate", 40, 200, 80)

chol = st.number_input("Cholesterol Level", 100, 400, 180)

bmi = st.number_input("BMI", 10.0, 60.0, 24.5)

if gender == "Female":
    gender = 0
elif gender == "Male":
    gender = 1
else:
    gender = 2

if st.button("Predict Disease"):

    data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Blood_Pressure": bp,
        "Heart_Rate": heart,
        "Cholesterol_Level": chol,
        "BMI": bmi
    }])

    pred = model.predict(data)

    disease = target_encoder.inverse_transform(pred)

    confidence = model.predict_proba(data).max() * 100

    st.success(f"Predicted Disease : {disease[0]}")

    st.info(f"Confidence : {confidence:.2f}%")

    st.markdown("---")

    st.subheader("General Health Advice")

    st.write("✔ Maintain a healthy diet")

    st.write("✔ Exercise regularly")

    st.write("✔ Drink enough water")

    st.write("✔ Consult a doctor for confirmation")

    st.warning("This prediction is for educational purposes only and is not a medical diagnosis.")