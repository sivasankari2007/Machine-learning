
import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as K

app = Flask(__name__)

# Load trained model
model = load_model(r"lung_prediction/lung_Cancer_Prediction.h5")

# Create folders
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Preprocess image
def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

# Grad-CAM
def generate_gradcam(model, img_array, layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(conv_outputs * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap[0], class_idx

# Heatmap overlay
def overlay_heatmap(img_path, heatmap, alpha=0.5):
    original_img = cv2.imread(img_path)
    original_img = cv2.resize(original_img, (224, 224))
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlayed_img = cv2.addWeighted(original_img, 1 - alpha, heatmap_colored, alpha, 0)
    return overlayed_img

# Recommendations
def get_recommendation(label):
    recommendations = {
        "Benign": "Benign lesion detected. Regular monitoring and follow-up with a physician are recommended.",
        "Malignant": "Malignant lesion detected. Please consult an oncologist immediately for further diagnosis and treatment.",
        "Normal": "No signs of lung cancer detected. Maintain a healthy lifestyle and consider regular checkups."
    }
    return recommendations.get(label, "")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded.")
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No file selected.")
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        img_array = preprocess_image(filepath)
        last_conv_layer = "block4_conv3"
        heatmap, predicted_class = generate_gradcam(model, img_array, last_conv_layer)
        overlayed_img = overlay_heatmap(filepath, heatmap)

        gradcam_path = os.path.join(RESULT_FOLDER, "gradcam_" + file.filename)
        cv2.imwrite(gradcam_path, overlayed_img)

        class_labels = ["Benign", "Malignant", "Normal"]
        predicted_label = class_labels[predicted_class]
        recommendation = get_recommendation(predicted_label)

        return render_template("index.html",
                               uploaded_img=filepath,
                               gradcam_img=gradcam_path,
                               prediction=predicted_label,
                               recommendation=recommendation)
    return render_template("index.html")
if __name__ == "__main__":
    app.run(debug=True)
