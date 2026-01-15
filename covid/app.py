import streamlit as st
import numpy as np
import cv2
import tensorflow as tf

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="COVID-19 X-ray Classification",
    page_icon="🫁",
    layout="centered"
)

st.title("🫁 COVID-19 X-ray Classification System")
st.write("Upload a chest X-ray image to predict the disease class")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("covid_cnn_model.h5")

try:
    model = load_model()
except:
    st.error("❌ Model file 'covid_cnn_model.h5' not found")
    st.stop()

# ⚠️ Must match training order
CLASS_NAMES = [
    "COVID",
    "Lung_Opacity",
    "Normal",
    "Viral Pneumonia"
]

IMAGE_SIZE = 128

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Invalid image file")
        st.stop()

    st.image(img, caption="Uploaded Image", channels="BGR", use_container_width=True)

    # ---------------- PREPROCESS ----------------
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img / 255.0
    img = img.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)

    # ---------------- PREDICTION ----------------
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100

    # ---------------- OUTPUT ----------------
    st.success(f"🧠 Prediction: **{CLASS_NAMES[class_index]}**")
    st.info(f"📊 Confidence: **{confidence:.2f}%**")
