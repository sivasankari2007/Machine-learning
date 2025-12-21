import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import pandas as pd
import time
import requests


st.set_page_config(layout="wide")

# -----------------------------
# Load model and scaler
# -----------------------------
model = load_model("phone_addiction/mobile_addiction_cnn_model.h5")
scaler = joblib.load("phone_addiction/scaler (1).pkl")
le_target = joblib.load("phone_addiction/le_target.pkl")

# -----------------------------
# Initialize MediaPipe
# -----------------------------
mp_face = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# Initialize session state
# -----------------------------
if "consent_given" not in st.session_state:
    st.session_state.consent_given = False
if "risk_history" not in st.session_state:
    st.session_state.risk_history = []

# -----------------------------
# Sidebar Navigation
# -----------------------------
page = st.sidebar.selectbox(
    "Go to",
    ["Home / Consent", "Live Detection", "Risk Dashboard", "Alerts", "Settings", "Live API Demo"]
)

# -----------------------------
# Page 1: Home / Consent
# -----------------------------
if page == "Home / Consent":
    st.title("📱 Mobile Addiction Monitoring")
    st.write("""
        **Project Description:**  
        Monitors mobile usage risk based on posture, eye focus, and usage metrics.
    """)
    consent = st.checkbox("I give consent for live monitoring")
    if st.button("Start Monitoring"):
        if consent:
            st.session_state.consent_given = True
            st.success("✅ Consent accepted. You can now go to Live Detection page.")
        else:
            st.error("⚠️ Please give consent to proceed.")

# -----------------------------
# Page 2: Live Detection (Webcam + CNN)
# -----------------------------
elif page == "Live Detection":
    if not st.session_state.consent_given:
        st.warning("⚠️ Please give consent first on the Home page.")
        st.stop()

    st.title("📷 Live Detection")
    frame_window = st.image([])
    risk_text = st.empty()
    metrics_text = st.empty()

    cap = cv2.VideoCapture(0)
    start_time = time.time()

    def calculate_risk(head_tilt, posture_score):
        score = head_tilt * 0.4 + (1 - posture_score) * 50
        if score < 20:
            return "Low", round(score,2)
        elif score < 40:
            return "Medium", round(score,2)
        else:
            return "High", round(score,2)

    stop_button = st.button("Stop Live Detection")

    while True:
        ret, frame = cap.read()
        if not ret or stop_button:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face.process(rgb)

        head_tilt = 0
        eye_focus = 1  # 0: up, 1: center, 2: down
        posture_score = 0.5

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            nose = face.landmark[1]
            chin = face.landmark[152]

            head_tilt = abs(nose.y - chin.y) * 100
            posture_score = round(1 - min(head_tilt / 100,1),2)
            eye_focus = 2 if nose.y > 0.55 else 0 if nose.y < 0.45 else 1

            # Draw landmarks
            for lm in face.landmark:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                cv2.circle(frame, (x,y), 1, (0,255,0), -1)

        # Predict risk using CNN
        X_input = np.array([[head_tilt, posture_score, eye_focus]])
        X_input_scaled = scaler.transform(X_input)
        pred = model.predict(X_input_scaled)
        pred_class = le_target.inverse_transform([np.argmax(pred)])[0]

        # Store history
        st.session_state.risk_history.append({"time": time.time()-start_time, "risk": pred_class, "posture_score": posture_score})

        # Update display
        frame_window.image(frame, channels="BGR")
        metrics_text.write(f"Head Tilt: {head_tilt:.2f}, Posture Score: {posture_score}, Eye Focus: {eye_focus}")
        risk_text.subheader(f"Current Risk Level: {pred_class}")

    cap.release()
    cv2.destroyAllWindows()

# -----------------------------
# Page 3: Risk Dashboard
# -----------------------------
elif page == "Risk Dashboard":
    st.title("📊 Risk Dashboard")
    if not st.session_state.risk_history:
        st.info("No data yet. Run Live Detection first.")
    else:
        df = pd.DataFrame(st.session_state.risk_history)
        df['time_sec'] = df['time']

        st.subheader("Line Chart: Risk vs Time")
        st.line_chart(df['posture_score'].values)

        st.subheader("Bar Chart: Daily Risk Count")
        st.bar_chart(df['risk'].value_counts())

        st.write("Average Posture Score:", df['posture_score'].mean())
        st.write("Recommendations:")
        st.write("- Take breaks if High risk appears frequently.")
        st.write("- Maintain upright posture.")

# -----------------------------
# Page 4: Alerts
# -----------------------------
elif page == "Alerts":
    st.title("⚠️ Alerts Page")
    if not st.session_state.risk_history:
        st.info("No alerts yet. Run Live Detection first.")
    else:
        df = pd.DataFrame(st.session_state.risk_history)
        high_risk_df = df[df['risk']=="High"]
        st.write("High Risk Log:")
        st.dataframe(high_risk_df)
        if not high_risk_df.empty:
            st.warning("⚠️ High risk detected! Take a break!")

# -----------------------------
# Page 5: Settings
# -----------------------------
elif page == "Settings":
    st.title("⚙️ Settings")
    st.write("Adjust sensitivity, risk thresholds, or reset data.")
    reset = st.button("Reset Risk History")
    if reset:
        st.session_state.risk_history = []
        st.success("✅ Risk history cleared.")

# -----------------------------
# Page 6: Live API Demo
# -----------------------------
elif page == "Live API Demo":
    st.title("🌐 Free Live API Demo")
    st.write("This uses a free public API as a placeholder for live data.")

    if st.button("Fetch Live API Data"):
        try:
            response = requests.get("https://api.publicapis.org/entries")  # Free placeholder API
            if response.status_code == 200:
                st.success("✅ API connected successfully")
                # Display 3 entries as demo
                entries = response.json()['entries'][:3]
                st.json(entries)
            else:
                st.error("API request failed")
        except Exception as e:
            st.error(f"API error: {e}")
