import streamlit as st
import pickle
import numpy as np

# Load model
with open("smart/smartphonepredict.pkl", "rb") as f:
    model = pickle.load(f)

st.title("📱 Mobile Price Prediction App")

# Get number of features expected by model
num_features = model.n_features_in_

st.write(f"🔢 Model expects {num_features} features")

inputs = []

# Create dynamic input boxes
for i in range(num_features):
    value = st.number_input(f"Feature {i+1}", value=0.0)
    inputs.append(value)

if st.button("Predict"):
    input_array = np.array(inputs).reshape(1, -1)
    prediction = model.predict(input_array)
    st.success(f"Predicted Price: ₹ {prediction[0]:.2f}")



