import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Page title
st.title("🧠 Brain Tumor Detection App")
st.write("Upload an MRI image to check whether a brain tumor is detected.")

# Load trained model
model = load_model("Brain_Tumor_Detection/brain_tumor_dataset.h5")

# Upload image
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
    prediction = model.predict(img_array)[0][0]

    st.write("### Prediction Result:")

    if prediction > 0.5:
        st.error("⚠️ Brain Tumor Detected")
        st.write(f"Confidence: {prediction:.2f}")
    else:
        st.success("✅ No Brain Tumor Detected")
        st.write(f"Confidence: {1 - prediction:.2f}")
