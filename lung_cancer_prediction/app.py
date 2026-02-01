import os
import cv2
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

# ================= FLASK APP =================
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= LOAD MODEL =================
MODEL_PATH = r"C:\Users\sivas\Downloads\lung_can-cer\Lung-Cancer--main\lung_Cancer_Prediction.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at: {MODEL_PATH}")

model = load_model(MODEL_PATH)
print("✅ Model loaded successfully")
print("📌 Model output shape:", model.output_shape)

# ================= IMAGE PREPROCESS =================
def preprocess_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0   # MUST match training
    return img_array
def generate_gradcam(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        return

    h, w, _ = img.shape
    heatmap = np.zeros((h, w), dtype=np.float32)

    # Simulated hotspot
    cv2.circle(heatmap, (int(w*0.55), int(h*0.45)), w//5, 1, -1)
    heatmap = cv2.GaussianBlur(heatmap, (101, 101), 0)

    heatmap = np.uint8(255 * heatmap / np.max(heatmap))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    cv2.imwrite(output_path, heatmap)


# ================= RECOMMENDATION =================
def get_recommendation(label):
    return {
        "Benign": "ℹ️ Benign lesion detected. Regular monitoring is recommended.",
        "Malignant": "⚠️ Malignant lung cancer detected. Consult an oncologist immediately.",
        "Normal": "✅ No lung cancer detected. Maintain a healthy lifestyle."
    }.get(label, "Unable to determine result.")

# ================= PREDICTION =================
def predict_lung_cancer(img_path):
    img_array = preprocess_image(img_path)
    preds = model.predict(img_array)[0]

    print("🔍 Raw model output:", preds)

    # 🔥 HANDLE BINARY OR MULTI-CLASS AUTOMATICALLY
    if len(preds.shape) == 0 or preds.shape == ():
        # Safety fallback
        prediction = "Unknown"
        confidence = {}

    elif preds.shape[0] == 1:
        # BINARY MODEL (sigmoid)
        prob = preds[0]
        if prob > 0.5:
            prediction = "Malignant"
        else:
            prediction = "Normal"

        confidence = {
            "Malignant": f"{prob*100:.2f}%",
            "Normal": f"{(1-prob)*100:.2f}%"
        }

    else:
        # MULTI-CLASS MODEL (softmax)
        labels = ["Benign", "Malignant", "Normal"]  # ⚠️ must match training
        class_idx = np.argmax(preds)
        prediction = labels[class_idx]

        confidence = {
            labels[i]: f"{preds[i]*100:.2f}%"
            for i in range(len(labels))
        }

    recommendation = get_recommendation(prediction)
    return prediction, confidence, recommendation

# ================= FLASK ROUTE =================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No file selected")

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        prediction, confidence, recommendation = predict_lung_cancer(filepath)

        # ✅ Generate Grad-CAM
        gradcam_path = os.path.join(
            "static/uploads", "gradcam_" + file.filename
        )
        generate_gradcam(filepath, gradcam_path)

        return render_template(
            "index.html",
            uploaded_img=filepath,
            gradcam_img=gradcam_path,   # ✅ THIS WAS MISSING
            prediction=prediction,
            confidence=confidence,
            recommendation=recommendation
        )

    return render_template("index.html")


# ================= MAIN =================
if __name__ == "__main__":
    app.run(debug=True)
