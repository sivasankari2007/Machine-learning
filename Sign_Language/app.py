import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

st.title("✋ Hand Gesture / Sign Language Recognition")

model = load_model("Sign_Language/hand_gesture_model.h5")
CLASSES = ['A','B','C','D','E']

uploaded_file = st.file_uploader(
    "Upload a hand gesture image",
    type=["jpg","png","jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = np.array(image)
    img = cv2.resize(img, (64,64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"Prediction: {CLASSES[class_id]}")
    st.info(f"Confidence: {confidence*100:.2f}%")
