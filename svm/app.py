# app.py
import streamlit as st
import numpy as np
import pickle

st.title("Student Performance Prediction App")

# Load model and scaler
with open('svm/svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('svm/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# User input
gender = st.selectbox("Gender", ["female", "male"])
race = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parent_edu = st.selectbox("Parental Level of Education", [
    "some high school", "high school", "some college", 
    "associate's degree", "bachelor's degree", "master's degree"
])
lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
test_prep = st.selectbox("Test Preparation Course", ["none", "completed"])

# ✅ Use sliders for scores (easy to increase/decrease)
math_score = st.slider("Math Score", min_value=0, max_value=100, value=50, step=1)
reading_score = st.slider("Reading Score", min_value=0, max_value=100, value=50, step=1)
writing_score = st.slider("Writing Score", min_value=0, max_value=100, value=50, step=1)

# Map input to same encoding used in training
gender_map = {"female":0, "male":1}
race_map = {"group A":0, "group B":1, "group C":2, "group D":3, "group E":4}
parent_edu_map = {"some high school":0, "high school":1, "some college":2, "associate's degree":3,
                  "bachelor's degree":4, "master's degree":5}
lunch_map = {"standard":0, "free/reduced":1}
test_prep_map = {"none":0, "completed":1}

input_data = np.array([[gender_map[gender], race_map[race], parent_edu_map[parent_edu],
                        lunch_map[lunch], test_prep_map[test_prep],
                        math_score, reading_score, writing_score]])

# Scale input
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]
    if prediction[0] == 1:
        st.success(f"The student is likely to PASS ✅ (Probability: {probability:.2f})")
    else:
        st.error(f"The stu


