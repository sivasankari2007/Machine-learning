import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
import requests
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Mobile Addiction Monitor", layout="wide")

# ==================================================
# Load Model & Preprocessing Objects
# ==================================================
model = load_model("mobile_addiction_cnn_model.h5")
scaler = joblib.load("")
le_target = joblib.load("le_target.pkl")

# ==================================================
# Session State Initialization (Backend Memory)
# ==================================================
if "consent" not in st.session_state:
    st.session_state.consent = False

if "risk_data" not in st.session_state:
    st.session_state.risk_data = []

if "sensitivity" not in st.session_state:
    st.session_state.sensitivity = 1.0

# ==================================================
# Sidebar Navigation (Frontend Routing)
# ==================================================
page = st.sidebar.radio(
    "Navigation",
    [
        "1️⃣ Home / Consent",
        "2️⃣ Live Detection",
        "3️⃣ Risk Dashboard",
        "4️⃣ Alerts",
        "5️⃣ Settings",
        "6️⃣ Live API Demo"
    ]
)

# ==================================================
# Backend: Risk Logic
# ==================================================
def calculate_risk(head_tilt, posture_score, eye_focus):
    score = (head_tilt * 0.4 + (1 - posture_score) * 50) * st.session_state.sensitivity

    if score < 20:
        return "Low"
    elif score < 40:
        return "Medium"
    else:
        return "High"

# ==================================================
# PAGE 1: Home / Consent
# ==================================================
if page == "1️⃣ Home / Consent":
    st.title("📱 CNN-Based Mobile Addiction Detection")

    st.write("""
    **Project Description (Simple):**  
    This system analyzes posture and usage behavior to predict mobile addiction risk
    using a CNN model.
    """)

    consent = st.checkbox("I give consent for monitoring")

    if st.button("Start Monitoring"):
        if consent:
            st.session_state.consent = True
            st.success("✅ Consent accepted")
        else:
            st.error("❌ Consent required")

# ==================================================
# PAGE 2: Live Detection (Demo – Backend Update)
# ==================================================
elif page == "2️⃣ Live Detection":
    if not st.session_state.consent:
        st.warning("⚠️ Please give consent first")
        st.stop()

    st.title("📷 Live Detection (Demo Mode)")

    st.info("Cloud version uses simulated live posture data")

    if st.button("Generate Live Frame"):
        head_tilt = np.random.randint(5, 60)
        posture_score = round(np.random.uniform(0.3, 0.9), 2)
        eye_focus = np.random.choice([0, 1, 2])

        X = np.array([[head_tilt, posture_score, eye_focus]])
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)
        risk = le_target.inverse_transform([np.argmax(pred)])[0]

        st.session_state.risk_data.append({
            "Time": time.strftime("%H:%M:%S"),
            "HeadTilt": head_tilt,
            "PostureScore": posture_score,
            "Risk": risk
        })

        st.metric("Head Tilt", head_tilt)
        st.metric("Posture Score", posture_score)
        st.subheader(f"⚠️ Risk Level: {risk}")

# ==================================================
# PAGE 3: Risk Dashboard (Frontend Analytics)
# ==================================================
elif page == "3️⃣ Risk Dashboard":
    st.title("📊 Risk Dashboard")

    if not st.session_state.risk_data:
        st.info("No data available")
    else:
        df = pd.DataFrame(st.session_state.risk_data)

        st.subheader("Risk Distribution")
        st.bar_chart(df["Risk"].value_counts())

        st.subheader("Posture Score Over Time")
        st.line_chart(df["PostureScore"])

        st.write("Average Posture Score:", round(df["PostureScore"].mean(), 2))
        st.dataframe(df)

# ==================================================
# PAGE 4: Alerts Page
# ==================================================
elif page == "4️⃣ Alerts":
    st.title("🚨 Alerts & Warnings")

    if not st.session_state.risk_data:
        st.info("No alerts yet")
    else:
        df = pd.DataFrame(st.session_state.risk_data)
        high_risk = df[df["Risk"] == "High"]

        if high_risk.empty:
            st.success("✅ No high-risk alerts")
        else:
            st.error("⚠️ High Risk Detected!")
            st.dataframe(high_risk)
            st.warning("💡 Recommendation: Take a break & correct posture")

# ==================================================
# PAGE 5: Settings Page (Backend Control)
# ==================================================
elif page == "5️⃣ Settings":
    st.title("⚙️ Settings")

    st.session_state.sensitivity = st.slider(
        "Risk Sensitivity",
        0.5, 2.0,
        st.session_state.sensitivity,
        0.1
    )

    if st.button("Reset All Data"):
        st.session_state.risk_data = []
        st.success("✅ Data reset completed")

# ==================================================
# PAGE 6: Live API Demo
# ==================================================
elif page == "6️⃣ Live API Demo":
    st.title("🌐 Free Live API Connection")

    st.write("Demonstrates external API usage")

    if st.button("Fetch API Data"):
        try:
            res = requests.get("https://api.publicapis.org/entries", timeout=5)
            if res.status_code == 200:
                st.success("API Connected Successfully")
                st.json(res.json()["entries"][:3])
            else:
                st.error("API request failed")
        except Exception as e:
            st.error(e)
