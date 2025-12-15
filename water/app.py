import streamlit as st
import pickle
import numpy as np

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Water Potability Prediction",
    page_icon="💧",
    layout="centered"
)

# -------------------------------
# Load model and scaler
# -------------------------------
try:
    with open("", "rb") as f:
        model, scaler = pickle.load(f)
except FileNotFoundError:
    st.error("❌ hardness_model.pkl not found. Please place it in the same folder as app.py")
    st.stop()

# -------------------------------
# App Title
# -------------------------------
st.title("💧 Water Potability Prediction")
st.write("Enter water quality parameters to predict if it is potable (safe to drink).")

# -------------------------------
# User Inputs
# -------------------------------
ph = st.number_input("pH value", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
Hardness = st.number_input("Hardness (mg/L)", min_value=0.0, max_value=500.0, value=150.0, step=1.0)
