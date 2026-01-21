import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from difflib import get_close_matches

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="HealthCare Chatbot", page_icon="🩺", layout="wide")

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("health_chatbot/disease_model.pkl", "rb"))
le = pickle.load(open("health_chatbot/label_encoder (2).pkl", "rb"))

# ---------------- LOAD DATA ----------------
train_df = pd.read_csv("health_chatbot/Training.csv")
train_df = train_df.loc[:, ~train_df.columns.str.contains("^Unnamed")]
symptom_list = list(train_df.columns[:-1])

desc_df = pd.read_csv("health_chatbot/symptom/symptom_Description.csv")
prec_df = pd.read_csv("health_chatbot/symptom/symptom_precaution.csv")

# Normalize column names
desc_df.columns = desc_df.columns.str.lower().str.strip()
prec_df.columns = prec_df.columns.str.lower().str.strip()

# ---------------- AUTO-DETECT DISEASE COLUMN ----------------
def find_disease_column(df):
    for col in df.columns:
        if "disease" in col or "prognosis" in col or "condition" in col:
            return col
    return df.columns[0]   # fallback (safe)

desc_disease_col = find_disease_column(desc_df)
prec_disease_col = find_disease_column(prec_df)

# ---------------- NLP ----------------
def extract_symptoms(text):
    text = text.lower()
    words = re.findall(r"\w+", text)
    found = []
    for w in words:
        match = get_close_matches(w, symptom_list, n=1, cutoff=0.75)
        if match:
            found.append(match[0])
    return list(set(found))

# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align:center;'>🩺 HealthCare Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-based disease prediction system using Machine Learning</p>", unsafe_allow_html=True)

left, right = st.columns([1.2, 1])

# ---------------- INPUT ----------------
with left:
    name = st.text_input("Your Name")
    age = st.number_input("Age", 0, 120, 0)
    gender = st.radio("Gender", ["Male", "Female", "Other"])
    symptoms_text = st.text_area("Describe your symptoms")
    days = st.number_input("Number of Days", 0, 30, 0)
    severity = st.slider("Severity (1–10)", 1, 10, 3)

    colA, colB = st.columns(2)
    submit = colB.button("Submit", type="primary")
    clear = colA.button("Clear")

# ---------------- OUTPUT ----------------
with right:
    if submit:
        if not symptoms_text.strip():
            st.error("❌ Please enter symptoms")
        else:
            symptoms = extract_symptoms(symptoms_text)

            if not symptoms:
                st.error("❌ No recognizable symptoms found")
            else:
                vec = np.zeros(len(symptom_list))
                for s in symptoms:
                    vec[symptom_list.index(s)] = 1

                pred = model.predict([vec])[0]
                prob = model.predict_proba([vec])[0]
                confidence = max(prob)
                disease = le.inverse_transform([pred])[0]

                # SAFE DESCRIPTION
                desc_row = desc_df[desc_df[desc_disease_col] == disease]
                description = desc_row.iloc[0, 1] if not desc_row.empty else "No description available."

                # SAFE PRECAUTIONS
                prec_row = prec_df[prec_df[prec_disease_col] == disease]
                precautions = prec_row.iloc[0, 1:].tolist() if not prec_row.empty else ["Consult doctor"]

                st.markdown(f"### 🧬 Predicted Disease: **{disease}**")
                st.markdown(f"🔵 **Confidence:** {round(confidence*100,2)}%")

                st.markdown("### 📘 About Disease:")
                st.write(description)

                st.markdown("### 🛡️ Precautions:")
                for p in precautions:
                    st.write("◦", p)

                st.markdown("💡 **Health is wealth.**")
                st.markdown(f"🧑‍⚕️ **Stay healthy, {name}!**")

    if clear:
        st.experimental_rerun()
