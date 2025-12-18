import streamlit as st
import pickle
import numpy as np

# Title
st.title("📺 Advertising Sales Prediction App")
st.write("Enter advertising budget to predict sales")

# Load model
@st.cache_resource
def load_model():
    with open("advertising/advertising_poly_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# User input
tv = st.number_input("TV Advertising Budget", min_value=0.0)
radio = st.number_input("Radio Advertising Budget", min_value=0.0)
newspaper = st.number_input("Newspaper Advertising Budget", min_value=0.0)

# Predict button
if st.button("Predict Sales"):
    input_data = np.array([[tv, radio, newspaper]])
    prediction = model.predict(input_data)

    st.success(f"📈 Predicted Sales: {prediction[0]:.2f}")

