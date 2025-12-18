import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

IMG_SIZE = 224

st.title("😊 Face Recognition App")

# Load model
model = load_model("face_recognition.h5")

# Load class names
with open("class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

uploaded_file = st.file_uploader(
    "Upload Face Image", type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width=250)

    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"🧑 Identified Person: **{class_names[class_index]}**")
    st.info(f"📊 Confidence: **{confidence*100:.2f}%**")
