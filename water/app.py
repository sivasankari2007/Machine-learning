import streamlit as st
import pickle
import numpy as np

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💼",
    layout="centered"
)

# -------------------------------
# Load the trained model
# -------------------------------
try:
    with open("", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("❌ salary_model.pkl not found. Please place it in the same folder as app.py")
    st.stop()

# -------------------------------
# App Title
# -------------------------------
st.title("💼 Salary Prediction App")
st.write("Predict your estimated salary based on years of experience.")

# -------------------------------
# User Input
# -------------------------------
experience = st.number_input(
    label="Years of Experience",
    min_value=0.0,
    max_value=50.0,
    value=1.0,
    step=0.1
)

# ----------------
