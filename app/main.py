"""
MAIN APPLICATION FILE
Location: templates/main.py
"""

from flask import Flask, jsonify, render_template, request
import sys
import os

# Add parent directory to path so we can import from app folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import from app folder
from app.predictor import predict_risk
from app.simulation import simulate
from app.recommendation import recommend
from app.alerts import send_alert
from app.xai import explain
from app.sustainability import sustainability
from app.soil import get_soil_data
from app.structure import get_structure_data
from app.weather_api import get_nasa_data
from app.weather_api_live import get_live_weather
from app.image_model import analyze_building

app = Flask(__name__, template_folder='.', static_folder='../static')

# ✅ Home route
@app.route("/")
def home():
    return render_template("index.html")

# ✅ Image Analysis
@app.route("/analyze_image", methods=["POST"])
def analyze_image_route():
    try:
        file = request.files["image"]
        path = "temp_upload.jpg"
        file.save(path)
        
        result = analyze_building(path)
        
        # Clean up
        if os.path.exists(path):
            os.remove(path)
            
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

# ✅ LIVE PREDICTION
@app.route("/predict_live", methods=["GET"])
def predict_live():
    try:
        print("🔥 API HIT - Starting prediction...")
        
        # -------------------------
        # 🌍 WEATHER (LIVE → NASA → DEFAULT)
        # -------------------------
        rainfall, temp, wind, humidity = None, None, None, None
        
        # Try live weather API first
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
        # 🌱 Soil + Structure
        # -------------------------
        try:
            soil = get_soil_data(16.54, 81.52)
            print("✅ Soil data:", soil)
        except Exception as e:
            print("❌ Soil error:", e)
            soil = {"type": "Clay Loam", "moisture": 45, "bearing": "Medium", "ph": 6.5}
        
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
            sim = int(simulate(data, rain_inc=20, temp_inc=2))
            print(f"📈 Simulation result: {sim}")
        except Exception as e:
            print(f"❌ Simulation error: {e}")
            sim = risk
        
        # Recommendation
        try:
            rec = recommend(risk)
            print(f"💡 Recommendation: {rec}")
        except Exception as e:
            print(f"❌ Recommendation error: {e}")
            rec = "Conduct structural inspection" if risk == 1 else "Routine maintenance"
        
        # Alert
        try:
            alert = send_alert(risk)
            print(f"🚨 Alert: {alert}")
        except Exception as e:
            print(f"❌ Alert error: {e}")
            alert = "HIGH RISK: Immediate action required" if risk == 1 else "Risk level normal"
        
        # Explanation (XAI)
        try:
            exp = explain(data)
            if not isinstance(exp, dict):
                exp = {"reasons": []}
            print(f"🧠 Explanation: {exp}")
        except Exception as e:
            print(f"❌ XAI error: {e}")
            exp = {"reasons": [f"High rainfall ({rainfall}mm) contributes to risk"]}
        
        # Sustainability
        try:
            sus = sustainability(risk)
            if not isinstance(sus, dict):
                sus = {"cost": "N/A", "durability": "N/A", "carbon": "N/A"}
            print(f"🌱 Sustainability: {sus}")
        except Exception as e:
            print(f"❌ Sustainability error: {e}")
            sus = {"cost": "Medium", "durability": "10 years", "carbon": "Moderate"}
        
        # -------------------------
        # ✅ FINAL RESPONSE
        # -------------------------
        response = {
            "success": True,
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
            "risk_level": "HIGH 🚨" if risk == 1 else "LOW ✅",
            "simulation": sim,
            "simulation_risk": "HIGH" if sim == 1 else "LOW",
            "recommendation": str(rec),
            "alert": str(alert),
            "explanation": exp,
            "sustainability": sus
        }
        
        print("✅ Prediction complete!")
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ FINAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
            "inputs": {}
        })

# ✅ Health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "message": "Server is running"})

# ✅ Run server
if __name__ == "__main__":
    print("="*50)
    print("🚀 AI Disaster Prediction System")
    print("="*50)
    print(f"📍 Access at: http://127.0.0.1:5000")
    print(f"📍 Health check: http://127.0.0.1:5000/health")
    print("="*50)
    app.run(debug=True, host='127.0.0.1', port=5000)