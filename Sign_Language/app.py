# app.py
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# ------------------ Load model ------------------
model = load_model("hand_gesture_model.h5")
st.title("✋ Hand Gesture / Sign Language Recognition")

# ------------------ Load Classes ------------------
CLASSES = ['A','B','C','D','E']  # Replace with your gesture folders names

# ------------------ Webcam ------------------
run = st.checkbox('Start Webcam')
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to grab frame")
        break

    # Flip for mirror view
    frame = cv2.flip(frame, 1)

    # Preprocess frame
    roi = cv2.resize(frame, (64,64))
    roi = roi / 255.0
    roi = np.expand_dims(roi, axis=0)

    # Prediction
    pred = model.predict(roi)
    class_idx = np.argmax(pred)
    confidence = np.max(pred)

    # Display prediction
    cv2.putText(frame, f"{CLASSES[class_idx]}: {confidence*100:.2f}%", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

    # Display in streamlit
    FRAME_WINDOW.image(frame, channels='BGR')

cap.release()
