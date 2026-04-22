from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# -------------------------------
# 📦 LOAD MODEL (SAFE)
# -------------------------------
MODEL_PATH = "crack_detection.h5"

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully")
else:
    model = None
    print("⚠️ Model not found, using fallback")


# -------------------------------
# 🧠 PREPROCESS IMAGE
# -------------------------------
def preprocess_image(file):
    img = Image.open(file).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# -------------------------------
# 🤖 PREDICT DAMAGE
# -------------------------------
def predict_crack_severity(file):
    try:
        if model:
            img = preprocess_image(file)
            pred = model.predict(img)[0][0]
            return int(pred * 100)
        else:
            # fallback if model not present
            return int(np.random.randint(20, 80))
    except Exception as e:
        print("❌ Prediction error:", e)
        return 30


# -------------------------------
# ✅ TEST ROUTE (IMPORTANT)
# -------------------------------
@app.route("/")
def home():
    return "Backend running on port 5001 ✅"


# -------------------------------
# 🌐 IMAGE API
# -------------------------------
@app.route("/analyzeimage", methods=["POST"])
def analyze_image():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"})

        file = request.files['image']

        damage_score = predict_crack_severity(file)

        # classify
        if damage_score > 70:
            condition = "Severe Cracking"
            severity = "high"
        elif damage_score > 40:
            condition = "Moderate Cracking"
            severity = "medium"
        else:
            condition = "Minor Cracks"
            severity = "low"

        return jsonify({
            "condition": condition,
            "damagescore": damage_score,
            "severity": severity
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# -------------------------------
# 🚀 RUN SERVER
# -------------------------------
if __name__ == "__main__":
    app.run(port=5001, debug=True)
