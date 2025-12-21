import streamlit as st
import numpy as np
import pandas as pd
import joblib
import time
import requests
from tensorflow.keras.models import load_model

st.set_page_config(layout="wide")

# ---------------------------------
# Load ML model & encoders
# ---------------------------------
model = load_model("phone_addiction/mobile_addiction_cnn_model.h5")
scaler = joblib.load("phone_addiction/scaler (1).pkl")
le_target = joblib.load("phone_addiction/le_target.pkl")

# ---------------------------------
# Session state
# ---------------------------------
if "consent_given" not in st.session_state:
    st.session_state.consent_given = False

if "risk_history" not in st.session_state:
    st.session_state.risk_history = []

# ---------------------------------
# Sidebar Navigation
# ---------------------------------
page = st.sidebar.selectbox(
    "Go to",
    [
        "Home / Consent",
        "Live Detection (Demo)",
        "Risk Dashboard",
        "Alerts",
        "Live API Demo"
    ]
)

# ---------------------------------
# Utility: Risk calculation
# ---------------------------------
def calculate_risk(head_tilt, posture_score):
    score = head_tilt * 0.4 + (1 - posture_score) * 50
    if score < 20:
        return "Low"
    elif score < 40:
        return "Medium"
    else:
        return "High"

# =================================
# PAGE 1: Home / Consent
# =================================
if page == "Home / Consent":
    st.title("📱 Mobile Addiction Monitoring System")

    st.write("""
    **Project Description (Simple):**  
    This project detects mobile addiction risk using posture and usage behavior.
    It uses a CNN model to classify risk levels.
    """)

    consent = st.checkbox("I give consent for monitoring")

    if st.button("Start Monitoring"):
        if consent:
            st.session_state.consent_given = True
            st.success("✅ Consent accepted. Go to Live Detection page.")
        else:
            st.error("❌ Please give consent to continue.")

# =================================
# PAGE 2: Live Detection (DEMO)
# =================================
elif page == "Live Detection (Demo)":

    if not st.session_state.consent_given:
        st.warning("⚠️ Please give consent first.")
        st.stop()

    st.title("📷 Live Detection (Demo Mode)")

    st.info("""
    Webcam & MediaPipe are **disabled on Streamlit Cloud**.
    This page simulates live posture data for demonstration.
    """)

    if st.button("Simulate Live Frame"):
        # Fake posture values (demo)
        head_tilt = np.random.randint(5, 60)
        posture_score = round(np.random.uniform(0.3, 0.9), 2)
        eye_focus = np.random.choice([0, 1, 2])

        X = np.array([[head_tilt, posture_score, eye_focus]])
        X_scaled = scaler.transform(X)

        pred = model.predict(X_scaled)
        risk = le_target.inverse_transform([np.argmax(pred)])[0]

        st.session_state.risk_history.append({
            "time": time.strftime("%H:%M:%S"),
            "risk": risk,
            "posture_score": posture_score
        })

        st.write("Head Tilt:", head_tilt)
        st.write("Posture Score:", posture_score)
        st.subheader(f"📊 Risk Level: {risk}")

# =================================
# PAGE 3: Risk Dashboard
# =================================
elif page == "Risk Dashboard":
    st.title("📊 Risk Dashboard")

    if not st.session_state.risk_history:
        st.info("No data available yet.")
    else:
        df = pd.DataFrame(st.session_state.risk_history)

        st.subheader("Risk Count")
        st.bar_chart(df["risk"].value_counts())

        st.subheader("Average Posture Score")
        st.write(df["posture_score"].mean())

        st.subheader("History Table")
        st.dataframe(df)

# =================================
# PAGE 4: Alerts
# =================================
elif page == "Alerts":
    st.title("⚠️ Alerts")

    if not st.session_state.risk_history:
        st.info("No alerts yet.")
    else:
        df = pd.DataFrame(st.session_state.risk_history)
        high_risk = df[df["risk"] == "High"]

        if high_risk.empty:
            st.success("No high-risk alerts 🎉")
        else:
            st.error("⚠️ High Risk Detected!")
            st.dataframe(high_risk)

# =================================
# PAGE 5: Live API Demo
# =================================
elif page == "Live API Demo":
    st.title("🌐 Free Live API Demo")

    st.write("Using a free public API (for academic demo).")

    if st.button("Fetch Live API Data"):
        try:
            r = requests.get("https://api.publicapis.org/entries", timeout=5)
            if r.status_code == 200:
                data = r.json()["entries"][:3]
                st.success("API Connected Successfully")
                st.json(data)
            else:
                st.error("API request failed")
        except Exception as e:
            st.error(f"API Error: {e}")
