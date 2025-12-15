import streamlit as st
import pickle
import numpy as np

# -------------------------------
# Load the trained model
# -------------------------------
with open("salary/salary_model.pkl", "rb") as file:
    model = pickle.load(file)

# -------------------------------
# App Title
# -------------------------------
st.title("💼 Salary Prediction App")
st.write("Enter years of experience to predict the estimated salary.")

# -------------------------------
# User Input
# -------------------------------
experience = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=1.0, step=0.1)

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("Predict Salary"):

