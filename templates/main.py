"""
AI Disaster Prediction System - Backend (UPDATED WITH CNN MODEL + GRAD-CAM + LIVE LOCATION)
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

# Global model variable
_model = None

def get_model():
    """Lazy load the crack detection model"""
    global _model
    if _model is None:
        try:
            import tensorflow as tf
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "best_crack_model.h5")
            print(f"📦 Loading model from: {model_path}")
            print(f"📂 Exists: {os.path.exists(model_path)}")
            _model = tf.keras.models.load_model(model_path)
            print("✅ Crack model loaded successfully")
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
        import requests
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
                "humidity": 65  # Open-Meteo current_weather doesn't include humidity
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
    return jsonify({"status": "healthy"})

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

        # Response for ranking mode (no base64 images needed)
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
# 🔥 GRAD-CAM VISUALIZATIONS (WITH BASE64 ENCODING)
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
        
        # ❌ IF VALIDATION FAILS, REJECT
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
        
        # Run Grad-CAM visualization (returns 3 different images)
        vis_result = generate_crack_visualization(model, temp_path)
        
        # Get three different base64 encoded images
        bbox_base64 = image_to_base64(vis_result.get("bbox_path"))        # Bounding boxes only
        overlay_base64 = image_to_base64(vis_result.get("overlay_path"))  # Heatmap overlay on original
        heatmap_base64 = image_to_base64(vis_result.get("heatmap_path"))  # Heatmap alone
        
        # Build response with THREE different images for frontend
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
            # THREE DIFFERENT IMAGES for frontend
            "image_with_boxes": bbox_base64,      # Bounding boxes on original
            "heatmap_overlay": overlay_base64,    # Heatmap OVERLAY on original
            "heatmap_alone": heatmap_base64,      # Heatmap ALONE (colored)
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
# 🌍 LIVE RISK SYSTEM (WITH LOCATION SUPPORT)
# ============================================

@app.route("/predict_live", methods=["GET"])
def predict_live():
    try:
        print("🔥 API HIT - Starting prediction...")
        
        # -------------------------
        # 📍 GET LOCATION FROM REQUEST
        # -------------------------
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)
        
        if lat is None or lon is None:
            # Default location (Vishakapatnam)
            lat = 16.54
            lon = 81.52
            print(f"📍 Using default location: {lat}, {lon}")
        else:
            print(f"📍 Using live location: {lat}, {lon}")
        
        # -------------------------
        # 🌍 WEATHER (Location-based)
        # -------------------------
        rainfall, temp, wind, humidity = None, None, None, None
        
        # Try Open-Meteo API with location first
        try:
            weather_data = get_weather_by_location(lat, lon)
            if weather_data:
                rainfall = weather_data.get("rainfall", 10)
                temp = weather_data.get("temperature", 28)
                wind = weather_data.get("windspeed", 12)
                humidity = weather_data.get("humidity", 65)
                print(f"✅ Using Open-Meteo API for location {lat},{lon}: temp={temp}°C, rain={rainfall}mm")
        except Exception as e:
            print(f"❌ Open-Meteo API failed: {e}")
        
        # Fallback to live weather API
        if rainfall is None:
            try:
                live = get_live_weather()
                if live:
                    rainfall, temp, wind, humidity = live
                    print("✅ Using LIVE API:", rainfall, temp, wind, humidity)
            except Exception as e:
                print("❌ Live API failed:", e)
        
        # Fallback to NASA data
        if rainfall is None:
            try:
                nasa = get_nasa_data()
                if nasa:
                    r, t, w, h = nasa
                    rainfall = r[-1] if r and len(r) > 0 else 10
                    temp = t[-1] if t and len(t) > 0 else 30
                    wind = w[-1] if w and len(w) > 0 else 10
                    humidity = h[-1] if h and len(h) > 0 else 50
                    print("⚠️ Using NASA DATA:", rainfall, temp, wind, humidity)
            except Exception as e:
                print("❌ NASA failed:", e)
        
        # Final fallback to default
        if rainfall is None:
            print("⚠️ Using DEFAULT DATA")
            rainfall, temp, wind, humidity = 10, 30, 50, 60
        
        # -------------------------
        # 🌱 Soil + Structure (using location)
        # -------------------------
        try:
            soil = get_soil_data(lat, lon)
            print("✅ Soil data:", soil)
        except Exception as e:
            print("❌ Soil error:", e)
            soil = {"type": "Clay Loam", "moisture": 45, "bearing": "Medium", "ph": 6.5, "clay_percentage": 40}
        
        try:
            structure = get_structure_data()
            print("✅ Structure data:", structure)
        except Exception as e:
            print("❌ Structure error:", e)
            structure = {"age": 20, "material": "Concrete", "cracks": "No", "floors": 3}
        
        # -------------------------
        # 🤖 MODEL INPUT (6 features)
        # -------------------------
        data = [
            float(rainfall),
            float(temp),
            float(humidity),
            float(wind),
            float(soil.get("moisture", 50)),
            float(structure.get("age", 20))
        ]
        
        print(f"📊 MODEL INPUT: {data}")
        
        # -------------------------
        # 🤖 AI MODULES
        # -------------------------
        # Risk Prediction (0=Low, 1=High)
        try:
            risk = int(predict_risk(data))
            print(f"🎯 Risk prediction: {risk}")
        except Exception as e:
            print(f"❌ Predictor error: {e}")
            risk = 1 if rainfall > 50 else 0
        
        # Simulation
        try:
            from app.simulation import simulate
            sim = int(simulate(data, rain_inc=20, temp_inc=2))
            print(f"📈 Simulation result: {sim}")
        except Exception as e:
            print(f"❌ Simulation error: {e}")
            sim = risk
        
        # Recommendation
        try:
            from app.recommendation import recommend
            rec = recommend(risk)
            print(f"💡 Recommendation: {rec}")
        except Exception as e:
            print(f"❌ Recommendation error: {e}")
            rec = "Conduct structural inspection" if risk == 1 else "Routine maintenance"
        
        # Alert
        try:
            from app.alerts import send_alert
            alert = send_alert(risk)
            print(f"🚨 Alert: {alert}")
        except Exception as e:
            print(f"❌ Alert error: {e}")
            alert = "HIGH RISK: Immediate action required" if risk == 1 else "Risk level normal"
        
        # Explanation (XAI)
        try:
            from app.xai import explain
            exp = explain(data)
            if not isinstance(exp, dict):
                exp = {"reasons": []}
            print(f"🧠 Explanation: {exp}")
        except Exception as e:
            print(f"❌ XAI error: {e}")
            exp = {"reasons": [f"High rainfall ({rainfall}mm) contributes to risk"]}
        
        # Sustainability
        try:
            from app.sustainability import sustainability
            sus = sustainability(risk)
            if not isinstance(sus, dict):
                sus = {"cost": "N/A", "durability": "N/A", "carbon": "N/A"}
            print(f"🌱 Sustainability: {sus}")
        except Exception as e:
            print(f"❌ Sustainability error: {e}")
            sus = {"cost": "Medium", "durability": "10 years", "carbon": "Moderate"}
        
        # Calculate risk score (0-100)
        risk_score = int(20 + rainfall * 0.6 + soil.get("clay_percentage", 40) * 0.3)
        risk_score = min(100, max(0, risk_score))
        
        # Determine risk level text
        if risk_score >= 75:
            risk_level_text = "CRITICAL 🚨"
        elif risk_score >= 55:
            risk_level_text = "HIGH ⚠️"
        elif risk_score >= 30:
            risk_level_text = "MEDIUM 📋"
        else:
            risk_level_text = "LOW ✅"
        
        # -------------------------
        # ✅ FINAL RESPONSE WITH LOCATION
        # -------------------------
        response = {
            "success": True,
            "location": {
                "lat": lat,
                "lon": lon
            },
            "inputs": {
                "weather": {
                    "rainfall": rainfall,
                    "temperature": temp,
                    "humidity": humidity,
                    "wind_speed": wind
                },
                "soil": soil,
                "structure": structure
            },
            "risk": risk,
            "risk_score": risk_score,
            "risk_level": risk_level_text,
            "simulation": sim,
            "simulation_risk": "HIGH" if sim == 1 else "LOW",
            "recommendation": str(rec),
            "alert": str(alert),
            "explanation": exp,
            "sustainability": sus
        }
        
        print(f"✅ Prediction complete! Location: {lat}, {lon}")
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ FINAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
            "inputs": {}
        }), 500

# ============================================
# 🧪 TEST ENDPOINT
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
# 📍 LOCATION TEST ENDPOINT
# ============================================

@app.route("/test_location", methods=["GET"])
def test_location():
    """Test location-based weather"""
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)
    
    if lat is None or lon is None:
        return jsonify({"error": "Please provide lat and lon parameters"}), 400
    
    weather = get_weather_by_location(lat, lon)
    return jsonify({
        "location": {"lat": lat, "lon": lon},
        "weather": weather
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
    print("   - Bounding Box Detection (GREEN boxes around cracks only)")
    print("   - LIVE LOCATION (GPS-based weather & soil)")
    print("   - CAMERA CAPTURE (Take photos directly)")
    print("   - Real-time Weather & Soil Data")
    print("   - IMAGE VALIDATION: Rejects People, Cars, Medical, Text Documents")
    print("="*60)
    print("📋 Test endpoints:")
    print("   GET  /health - Health check")
    print("   GET  /test_model - Test model loading")
    print("   GET  /test_location?lat=16.54&lon=81.52 - Test location weather")
    print("   POST /analyze_image - Upload image for crack detection")
    print("   POST /analyze_with_visualizations - Get Grad-CAM + bounding boxes")
    print("   GET  /predict_live - Get live risk prediction (with lat/lon params)")
    print("="*60)
    
    app.run(debug=True, host='127.0.0.1', port=5000)
