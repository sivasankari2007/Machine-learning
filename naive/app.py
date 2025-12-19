# app.py
import streamlit as st
import numpy as np
import pickle

# White-page clean layout
st.set_page_config(page_title="Student Prediction", layout="centered")
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Load model
with open('naive/naive_bayes_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Sliders for easy value increase
age = st.slider("Age", min_value=1, max_value=100, value=17, step=1)
attendance = st.slider("Attendance (%)", min_value=0, max_value=100, value=90, step=1)

# Categorical inputs
video_games = st.selectbox("Video Games", ["Yes", "No"])
tution = st.selectbox("Tution", ["Yes", "No"])
health = st.selectbox("Health", ["Good", "Bad"])
stress = st.selectbox("Stress", ["Low", "High"])
daily_work = st.selectbox("Daily Work", ["Yes", "No"])
self_study = st.selectbox("Self Study", ["Yes", "No"])

# Mapping categorical to numeric
map_yes_no = {"Yes":1, "No":0}
map_health = {"Good":1, "Bad":0}
map_stress = {"Low":0, "High":1}

input_data = np.array([[age, attendance,
                        map_yes_no[video_games],
                        map_yes_no[tution],
                        map_health[health],
                        map_stress[stress],
                        map_yes_no[daily_work],
                        map_yes_no[self_study]]])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)[0][1]
    if prediction[0]==1:
        st.success(f"PASS ✅ Probability: {prob:.2f}")
    else:
        st.error(f"FAIL ❌ Probability: {prob:.2f}")




