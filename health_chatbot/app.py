import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from difflib import get_close_matches

# Load model
model = pickle.load(open("health_chatbot/disease_model.pkl", "rb"))
le = pickle.load(open("health_chatbot/label_encoder (2).pkl", "rb"))

# Load datasets
training_data = pd.read_csv("dataset/Training.csv")
training_data = training_data.loc[:, ~training_data.columns.str.contains("^Unnamed")]
symptoms_list = list(training_data.columns[:-1])

desc_df = pd.read_csv("symptom/symptom_Description.csv")
prec_df = pd.read_csv("symptom/symptom_precaution.csv")

# NLP Symptom Extraction
def extract_symptoms(text):
    text = text.lower()
    words = re.findall(r'\w+', text)
    detected = []

    for word in words:
        match = get_close_matches(word, symptoms_list, n=1, cutoff=0.75)
        if match:
            detected.append(match[0])
    return list(set(detected))

# Streamlit UI
st.set_page_config(page_title="Healthcare Chatbot", page_icon="🩺")
st.title("🩺 AI Healthcare Chatbot")
st.write("Tell me your symptoms in simple English 👇")

user_input = st.text_input("Enter your symptoms:")

if user_input:
    symptoms = extract_symptoms(user_input)

    if len(symptoms) == 0:
        st.error("❌ No symptoms detected. Please try again.")
    else:
        st.success(f"Detected Symptoms: {', '.join(symptoms)}")

        # Create input vector
        input_vector = np.zeros(len(symptoms_list))
        for symptom in symptoms:
            input_vector[symptoms_list.index(symptom)] = 1

        # Prediction
        prediction = model.predict([input_vector])[0]
        probabilities = model.predict_proba([input_vector])[0]
        confidence = max(probabilities)

        disease = le.inverse_transform([prediction])[0]

        # Get description
        desc = desc_df[desc_df["Disease"] == disease]["Description"].values[0]

        # Get precautions
        prec = prec_df[prec_df["Disease"] == disease].values.tolist()[0][1:]

        st.markdown("---")
        st.subheader("🧠 Prediction Result")
        st.write(f"**Disease:** {disease}")
        st.write(f"**Confidence:** {round(confidence * 100, 2)}%")

        st.subheader("📖 Description")
        st.write(desc)

        st.subheader("🛡️ Precautions")
        for p in prec:
            st.write("✔️", p)

        st.subheader("💡 Health Tip")
        st.info("Drink plenty of water, take proper rest, and consult a doctor if symptoms persist.")

        st.subheader("🌱 Motivation")
        st.success("Your health is your wealth. Take care of yourself!")
