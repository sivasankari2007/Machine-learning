import streamlit as st
import pickle
import numpy as np

# -------------------------------
# Load the trained model
# -------------------------------
with open("diabetes/diabetes_model.pkl", "rb") as file:
    model = pickle.load(file)

# -------------------------------
# App Title
# -------------------------------
st.title("🩺 Diabetes Prediction App By Sankari")
st.write("Enter your details to predict the likelihood of diabetes.")

# -------------------------------
# User Inputs
# -------------------------------
age = st.number_input("Age", min_value=1, max_value=120, value=30)
mass = st.number_input("Body Mass (BMI)", min_value=10.0, max_value=100.0, value=25.0)
insu = st.number_input("Insulin Level", min_value=0.0, max_value=900.0, value=100.0)
plas = st.number_input("Plasma Glucose Concentration", min_value=0.0, max_value=300.0, value=120.0)

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Predict"):
    input_data = np.array([[age, mass, insu, plas]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # probability of class 1 (diabetes)

    if prediction == 1:
        st.error(f"The model predicts DIABETES with probability {probability:.2f}")
    else:
        st.success(f"The model predicts NO DIABETES with probability {1 - probability:.2f}")




