import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Alzheimer’s Stage Detection",
    page_icon="🧠",
    layout="centered"
)


st.markdown(
    "<h1 style='text-align: center; color: #4B8BBE;'>🧠 Alzheimer’s Disease Stage Detection</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>Upload an MRI image to predict the Alzheimer’s stage</p>",
    unsafe_allow_html=True
)

st.divider()

@st.cache_resource
def load_trained_model():
    return load_model("Alzhimer/alzheimer_model.h5")

model = load_trained_model()

# Class labels (VERY IMPORTANT: same order as training)
class_names = [
    "Mild Demented",
    "Moderate Demented",
    "Non Demented",
    "Very Mild Demented"
]

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📤 Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image_display = Image.open(uploaded_file)

    st.image(
        image_display,
        caption="Uploaded MRI Image",
        use_column_width=True
    )

    # ---------------- PREPROCESS ----------------
    img = image_display.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # ---------------- PREDICTION ----------------
    predictions = model.predict(img_array)[0]
    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = predictions[predicted_index] * 100

    st.divider()

    # ---------------- RESULT ----------------
    st.markdown(" 🧪 Prediction Result")

    if predicted_class == "Non Demented":
        st.success(f"✅ **{predicted_class}**")
    else:
        st.warning(f"⚠️ **{predicted_class}**")

    st.markdown(f"### 🔍 Confidence: **{confidence:.2f}%**")

    # ---------------- PROBABILITY BAR ----------------
    st.markdown("## 📊 Stage-wise Confidence")

    for i, label in enumerate(class_names):
        st.write(f"**{label}**")
        st.progress(float(predictions[i]))

# ---------------- FOOTER ----------------
st.divider()
st.markdown(
    "<small>⚠️ This application is for educational purposes only and not a medical diagnosis.</small>",
    unsafe_allow_html=True
)
