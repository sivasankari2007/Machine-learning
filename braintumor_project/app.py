import os
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as K

# Reduce TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

st.set_page_config(page_title="Lung Cancer Detection", layout="centered")

st.title("🫁 Lung Cancer Detection using Deep Learning")
st.write("Upload a CT scan image to predict lung cancer and view Grad-CAM visualization.")

# ---------------- Load Model ----------------
@st.cache_resource
def load_trained_model():
    return load_model("Lung_Model.h5")

model = load_trained_model()

# ---------------- Get Last Conv Layer ----------------
def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found.")

LAST_CONV_LAYER = get_last_conv_layer(model)

# ---------------- Preprocess ----------------
def preprocess_image(img):
    img = img.resize((224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img / 255.0

# ---------------- Grad-CAM ----------------
def generate_gradcam(model, img_array, layer_name):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap[0].numpy(), int(class_idx)

# ---------------- Overlay ----------------
def overlay_heatmap(original_img, heatmap):
    img = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

# ---------------- Recommendation ----------------
def get_recommendation(label):
    data = {
        "Benign": "🟢 Benign lesion detected. Regular monitoring advised.",
        "Malignant": "🔴 Malignant lesion detected. Consult oncologist immediately.",
        "Normal": "✅ No lung cancer detected. Maintain a healthy lifestyle."
    }
    return data[label]

# ---------------- UI ----------------
uploaded_file = st.file_uploader("Upload CT Scan Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    from PIL import Image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess_image(img)
    heatmap, pred_class = generate_gradcam(model, img_array, LAST_CONV_LAYER)
    gradcam_img = overlay_heatmap(img, heatmap)

    labels = ["Benign", "Malignant", "Normal"]
    prediction = labels[pred_class]

    st.subheader("🧪 Prediction Result")
    st.success(f"Prediction: **{prediction}**")
    st.info(get_recommendation(prediction))

    st.subheader("🔥 Grad-CAM Visualization")
    st.image(gradcam_img, channels="BGR", use_column_width=True)
