import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import json

st.title("✋ Hand Gesture / Sign Language Recognition")

# Load model
model = load_model("Sign_Language/hand_gesture_model.h5")

# Load class names
with open("Sign_Language/class_names.json", "r") as f:
    CLASSES = json.load(f)

uploaded_file = st.file_uploader(
    "Upload a hand gesture image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = np.array(image)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_id = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    st.success(f"Prediction: {CLASSES[class_id]}")
    st.info(f"Confidence: {confidence*100:.2f}%")
