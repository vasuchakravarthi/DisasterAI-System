"""
AI Disaster Prediction System - Backend (WITH AUTO MODEL DOWNLOAD FROM GOOGLE DRIVE)
Location: templates/main.py
"""

from flask import Flask, jsonify, render_template, request
import sys
import os
import traceback
from werkzeug.utils import secure_filename
import base64
import cv2
import numpy as np
import requests

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from app.predictor import predict_risk
from app.soil import get_soil_data
from app.structure import get_structure_data
from app.weather_api import get_nasa_data
from app.weather_api_live import get_live_weather
from app.image_model import analyze_building_safe
from app.image_validator import validate_structural_image
from app.gradcam import generate_crack_visualization
from app.crack_predictor import predict_crack

app = Flask(__name__, template_folder='.', static_folder='../static')

# Config
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure upload folder exists
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ============================================
# 🔥 MODEL AUTO-DOWNLOAD FROM GOOGLE DRIVE
# ============================================

# Model path
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_crack_model.h5")

# ✅ YOUR GOOGLE DRIVE FILE ID (extracted from your link)
GOOGLE_DRIVE_FILE_ID = "1yBPrgoF2fCvyYiDRaFz5ZN7gaJ9XVp6_"

def download_model_from_drive():
    """Download model from Google Drive if not exists"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if not os.path.exists(MODEL_PATH):
        print("="*60)
        print("📥 MODEL NOT FOUND. DOWNLOADING FROM GOOGLE DRIVE...")
        print("="*60)
        
        try:
            # Try using gdown first (best for Google Drive)
            try:
                import gdown
                url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
                print(f"📂 Downloading from: {url}")
                gdown.download(url, MODEL_PATH, quiet=False)
                if os.path.exists(MODEL_PATH):
                    file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
                    print(f"✅ Model downloaded successfully! Size: {file_size:.2f} MB")
                    return True
            except ImportError:
                print("⚠️ gdown not installed, trying alternative method...")
            
            # Fallback: Use requests with Google Drive direct download
            print("📥 Using requests fallback method...")
            session = requests.Session()
            
            # Get confirmation token
            response = session.get(f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}", stream=True)
            
            # Handle Google Drive's virus scan warning
            confirm_token = None
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    confirm_token = value
                    break
            
            if confirm_token:
                download_url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}&confirm={confirm_token}"
            else:
                download_url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
            
            # Download file
            response = session.get(download_url, stream=True)
            total_size = 0
            
            with open(MODEL_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        total_size += len(chunk)
            
            file_size_mb = total_size / (1024 * 1024)
            print(f"✅ Model downloaded successfully! Size: {file_size_mb:.2f} MB")
            return True
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    else:
        file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"✅ Model already exists at {MODEL_PATH} (Size: {file_size:.2f} MB)")
        return True

# Download model on startup
print("\n🔍 Checking for model file...")
if download_model_from_drive():
    print("✅ Model ready for loading")
else:
    print("⚠️ WARNING: Could not download model automatically!")
    print("Please manually upload best_crack_model.h5 to the 'models' folder")

# Global model variable
_model = None

def get_model():
    """Lazy load the crack detection model"""
    global _model
    if _model is None:
        try:
            import tensorflow as tf
            
            if os.path.exists(MODEL_PATH):
                print(f"📦 Loading model from: {MODEL_PATH}")
                _model = tf.keras.models.load_model(MODEL_PATH)
                print("✅ Crack model loaded successfully!")
                
                # Print model info
                print(f"   Input shape: {_model.input_shape}")
                print(f"   Output shape: {_model.output_shape}")
            else:
                print(f"❌ Model not found at {MODEL_PATH}")
                _model = None
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            _model = None
    return _model

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image_path):
    """Convert image file to base64 string"""
    if image_path and os.path.exists(image_path):
        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    return None

def get_weather_by_location(lat, lon):
    """Get weather data using Open-Meteo API for specific coordinates"""
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&daily=rain_sum&timezone=auto"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if data.get('current_weather'):
            current = data['current_weather']
            rainfall = data.get('daily', {}).get('rain_sum', [0])[0] or 0
            return {
                "temperature": current.get('temperature', 25),
                "rainfall": rainfall,
                "windspeed": current.get('windspeed', 10),
                "humidity": 65
            }
    except Exception as e:
        print(f"Weather API error: {e}")
    return None

# ============================================
# ROUTES
# ============================================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health_check():
    model_status = "loaded" if get_model() is not None else "not loaded"
    model_exists = os.path.exists(MODEL_PATH)
    return jsonify({
        "status": "healthy",
        "model": model_status,
        "model_file_exists": model_exists,
        "model_path": MODEL_PATH
    })

@app.route("/model_status", methods=["GET"])
def model_status():
    """Check if model is available"""
    model = get_model()
    model_exists = os.path.exists(MODEL_PATH)
    file_size = None
    if model_exists:
        file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    
    return jsonify({
        "model_loaded": model is not None,
        "model_file_exists": model_exists,
        "model_path": MODEL_PATH,
        "file_size_mb": file_size
    })

# ============================================
# 🔥 SIMPLE IMAGE ANALYSIS (FOR RANKING MODE)
# ============================================

@app.route("/analyze_image", methods=["POST"])
def analyze_image():
    temp_path = None

    try:
        print("📸 Image received for analysis")

        if "image" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        # Save image
        filename = secure_filename(file.filename)
        temp_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(temp_path)

        # ============================================
        # 🔥 STEP 1: VALIDATE IMAGE
        # ============================================
        print(f"🔍 Validating image: {temp_path}")
        is_valid, validation_message, validation_details, warning = validate_structural_image(temp_path)
        
        if not is_valid:
            print(f"❌ Validation REJECTED: {validation_message}")
            return jsonify({
                "success": False,
                "error": True,
                "condition": "INVALID IMAGE",
                "damage_score": 0,
                "severity": "INVALID",
                "message": validation_message,
                "validation_details": validation_details
            }), 400

        print(f"✅ Validation PASSED: {validation_message}")
        if warning:
            print(f"⚠️ Warning: {warning}")

        # ============================================
        # 🔥 STEP 2: RUN PREDICTION
        # ============================================
        result = predict_crack(temp_path)

        if result.get("error"):
            return jsonify(result), 400

        # Response for ranking mode
        response = {
            "success": True,
            "condition": result.get("status"),
            "damage_score": result.get("damage_score"),
            "severity": result.get("severity"),
            "crack_detected": result.get("crack_detected"),
            "confidence": result.get("confidence"),
            "recommendation": result.get("recommendation"),
            "crack_count": result.get("crack_tiles", 0),
            "model": result.get("model_source"),
            "validation_message": validation_message,
            "warning": warning,
            "error": False
        }

        print(f"✅ Analysis result: Damage Score={response['damage_score']}%")
        return jsonify(response)

    except Exception as e:
        print("❌ Error:", e)
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": True,
            "message": str(e)
        }), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

# ============================================
# 🔥 GRAD-CAM VISUALIZATIONS
# ============================================

@app.route("/analyze_with_visualizations", methods=["POST"])
def analyze_with_visualizations():
    """Analyze image and return Grad-CAM heatmap + bounding boxes as BASE64"""
    temp_path = None
    try:
        print("📸 Generating Grad-CAM visualizations")

        if "image" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["image"]
        temp_path = "temp_vis.jpg"
        file.save(temp_path)

        # ============================================
        # 🔥 STEP 1: VALIDATE IMAGE
        # ============================================
        is_valid, validation_message, validation_details, warning = validate_structural_image(temp_path)
        
        if not is_valid:
            print(f"❌ Validation REJECTED: {validation_message}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return jsonify({
                "success": False,
                "error": True,
                "message": validation_message,
                "validation_details": validation_details
            }), 400

        print(f"✅ Validation PASSED: {validation_message}")
        if warning:
            print(f"⚠️ Warning: {warning}")

        # ============================================
        # 🔥 STEP 2: GET MODEL AND GENERATE VISUALIZATIONS
        # ============================================
        model = get_model()
        
        if model is None:
            return jsonify({
                "success": False,
                "error": True,
                "message": "Model not loaded. Please check model file."
            }), 500
        
        # Run crack prediction
        prediction_result = predict_crack(temp_path)
        
        # Run Grad-CAM visualization
        vis_result = generate_crack_visualization(model, temp_path)
        
        # Get base64 encoded images
        bbox_base64 = image_to_base64(vis_result.get("bbox_path"))
        overlay_base64 = image_to_base64(vis_result.get("overlay_path"))
        heatmap_base64 = image_to_base64(vis_result.get("heatmap_path"))
        
        # Build response
        result = {
            "success": True,
            "damage_score": prediction_result.get("damage_score", 0),
            "status": prediction_result.get("status", "UNKNOWN"),
            "severity": prediction_result.get("severity", "LOW"),
            "crack_count": prediction_result.get("crack_tiles", 0),
            "total_tiles": prediction_result.get("total_tiles", 0),
            "crack_probability": vis_result.get("crack_probability", prediction_result.get("crack_probability", 0)),
            "confidence": vis_result.get("confidence", prediction_result.get("confidence", 0)),
            "num_cracks": vis_result.get("num_cracks", 0),
            "image_with_boxes": bbox_base64,
            "heatmap_overlay": overlay_base64,
            "heatmap_alone": heatmap_base64,
            "recommendation": prediction_result.get("recommendation", "Schedule structural inspection"),
            "validation_message": validation_message,
            "validation_passed": True
        }
        
        if warning:
            result["warning"] = warning
        
        # Clean up temporary files
        for f in [vis_result.get("bbox_path"), vis_result.get("overlay_path"), 
                  vis_result.get("heatmap_path")]:
            if f and os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass
        
        print(f"✅ Visualizations generated: Crack Count={result.get('crack_count', 0)}")
        return jsonify(result)

    except Exception as e:
        print("❌ Visualization error:", e)
        traceback.print_exc()
        return jsonify({
            "success": False, 
            "error": True,
            "message": str(e)
        }), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

# ============================================
# 🌍 LIVE RISK SYSTEM (WITH LOCATION)
# ============================================

@app.route("/predict_live", methods=["GET"])
def predict_live():
    try:
        print("🔥 API HIT - Starting prediction...")
        
        # Get location from request
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)
        
        if lat is None or lon is None:
            lat = 16.54
            lon = 81.52
            print(f"📍 Using default location: {lat}, {lon}")
        else:
            print(f"📍 Using live location: {lat}, {lon}")
        
        # Get weather using location
        rainfall, temp, wind, humidity = None, None, None, None
        
        try:
            weather_data = get_weather_by_location(lat, lon)
            if weather_data:
                rainfall = weather_data.get("rainfall", 10)
                temp = weather_data.get("temperature", 28)
                wind = weather_data.get("windspeed", 12)
                humidity = weather_data.get("humidity", 65)
                print(f"✅ Weather fetched for location: temp={temp}°C, rain={rainfall}mm")
        except Exception as e:
            print(f"❌ Weather API failed: {e}")
        
        # Fallback values
        if rainfall is None:
            rainfall, temp, wind, humidity = 10, 28, 12, 65
        
        # Soil data
        try:
            soil = get_soil_data(lat, lon)
        except Exception as e:
            print(f"❌ Soil error: {e}")
            soil = {"type": "Clay Loam", "moisture": 45, "clay_percentage": 40}
        
        # Structure data
        try:
            structure = get_structure_data()
        except Exception as e:
            print(f"❌ Structure error: {e}")
            structure = {"age": 20}
        
        # Calculate risk score
        risk_score = int(20 + rainfall * 0.6 + soil.get("clay_percentage", 40) * 0.3)
        risk_score = min(100, max(0, risk_score))
        
        # Determine risk level
        if risk_score >= 75:
            risk_level = "CRITICAL 🚨"
        elif risk_score >= 55:
            risk_level = "HIGH ⚠️"
        elif risk_score >= 30:
            risk_level = "MEDIUM 📋"
        else:
            risk_level = "LOW ✅"
        
        response = {
            "success": True,
            "location": {"lat": lat, "lon": lon},
            "risk_score": risk_score,
            "risk_level": risk_level,
            "weather": {
                "temperature": temp,
                "rainfall": rainfall,
                "humidity": humidity,
                "wind_speed": wind
            },
            "soil": soil,
            "structure": structure,
            "recommendation": "Schedule immediate inspection" if risk_score > 50 else "Routine maintenance recommended"
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

# ============================================
# TEST ENDPOINTS
# ============================================

@app.route("/test_model", methods=["GET"])
def test_model():
    """Test if model is loaded correctly"""
    model = get_model()
    if model is None:
        return jsonify({"status": "error", "message": "Model not loaded"})
    
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "model_input_shape": str(model.input_shape),
        "model_output_shape": str(model.output_shape)
    })

# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    # Pre-load model on startup
    get_model()
    
    print("="*60)
    print("🚀 AI DISASTER PREDICTION SYSTEM")
    print("="*60)
    print("👉 http://127.0.0.1:5000")
    print("="*60)
    print("✅ Features:")
    print("   - CNN Crack Detection (97.23% accuracy)")
    print("   - Grad-CAM Heatmap Visualization")
    print("   - Auto-download model from Google Drive")
    print("   - Live Location & Weather")
    print("="*60)
    print("📋 Test endpoints:")
    print("   GET  /health - Health check")
    print("   GET  /model_status - Check model status")
    print("   GET  /test_model - Test model loading")
    print("="*60)
    
    app.run(debug=True, host='127.0.0.1', port=5000)
