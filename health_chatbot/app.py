import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from difflib import get_close_matches

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="HealthCare Chatbot (ML + API)",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = pickle.load(open("health_chatbot/disease_model.pkl", "rb"))
le = pickle.load(open("health_chatbot/label_encoder (2).pkl", "rb"))

# -----------------------------
# Load Datasets
# -----------------------------
train_df = pd.read_csv("dataset/Training.csv")
train_df = train_df.loc[:, ~train_df.columns.str.contains("^Unnamed")]
symptom_list = list(train_df.columns[:-1])

desc_df = pd.read_csv("symptom/symptom_Description.csv")
prec_df = pd.read_csv("symptom/symptom_precaution.csv")

# Normalize column names (IMPORTANT)
desc_df.columns = desc_df.columns.str.strip().str.lower()
prec_df.columns = prec_df.columns.str.strip().str.lower()

disease_col_desc = "disease" if "disease" in desc_df.columns else "prognosis"
disease_col_prec = "disease" if "disease" in prec_df.columns else "prognosis"

# -----------------------------
# NLP Symptom Extraction
# -----------------------------
def extract_symptoms(text):
    text = text.lower()
    words = re.findall(r"\w+", text)
    found = []
    for w in words:
        match = get_close_matches(w, symptom_list, n=1, cutoff=0.75)
        if match:
            found.append(match[0])
    return list(set(found))

# -----------------------------
# UI
# -----------------------------
st.markdown(
    "<h1 style='text-align:center;'>🩺 HealthCare Chatbot (ML + Live API)</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>AI-based disease prediction using Random Forest</p>",
    unsafe_allow_html=True
)

col1, col2 = st.columns([1, 1.2])

# -----------------------------
# INPUT SECTION
# -----------------------------
with col1:
    name = st.text_input("Your Name")
    age = st.number_input("Age", min_value=0, max_value=120, value=0)
    gender = st.radio("Gender", ["Male", "Female", "Other"])
    symptoms_text = st.text_area("Describe your symptoms")
    days = st.number_input("Number of Days", min_value=0, value=0)
    severity = st.slider("Severity (1–10)", 1, 10, 1)

    submit = st.button("Submit", type="primary")
    clear = st.button("Clear")

# -----------------------------
# OUTPUT SECTION
# -----------------------------
with col2:
    st.subheader("🧾 Diagnosis Result")

    if submit:
        if symptoms_text.strip() == "":
            st.error("❌ Please enter your symptoms.")
        else:
            symptoms = extract_symptoms(symptoms_text)

            if not symptoms:
                st.error("❌ No recognizable symptoms found.")
            else:
                st.success(f"Detected Symptoms: {', '.join(symptoms)}")

                input_vector = np.zeros(len(symptom_list))
                for s in symptoms:
                    input_vector[symptom_list.index(s)] = 1

                pred = model.predict([input_vector])[0]
                probs = model.predict_proba([input_vector])[0]
                confidence = max(probs)

                disease = le.inverse_transform([pred])[0]

                desc_row = desc_df[desc_df[disease_col_desc] == disease]
                description = (
                    desc_row["description"].values[0]
                    if not desc_row.empty
                    else "No description available."
                )

                prec_row = prec_df[prec_df[disease_col_prec] == disease]
                precautions = (
                    prec_row.values.tolist()[0][1:]
                    if not prec_row.empty
                    else ["Consult a doctor"]
                )

                st.markdown("---")
                st.markdown(f"### 🦠 Predicted Disease: **{disease}**")
                st.markdown(f"**Confidence:** {round(confidence * 100, 2)}%")

                st.markdown("### 📖 Description")
                st.write(description)

                st.markdown("### 🛡️ Precautions")
                for p in precautions:
                    st.write("✔️", p)

                st.markdown("### 💡 Health Tip")
                st.info("Drink plenty of water, take rest, and avoid self-medication.")

                st.markdown("### 🌱 Motivation")
                st.success("Your health is your greatest wealth. Take care!")

    if clear:
        st.experimental_rerun()
