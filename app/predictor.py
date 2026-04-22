# predictor.py - FIXED VERSION
import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "models", "model.pkl")

# Load model
try:
    model = joblib.load(model_path)
    print("✅ Model loaded successfully from:", model_path)
except Exception as e:
    print(f"⚠️ Could not load model: {e}")
    model = None

def predict_risk(data):
    """
    Predict risk based on input features
    data: list of 6 features [rainfall, temp, humidity, wind, soil_moisture, building_age]
    Returns: 1 for high risk, 0 for low risk
    """
    if model is None:
        # Fallback logic if model not available
        rainfall, temp, humidity, wind, soil_moisture, building_age = data
        risk_score = 0
        if rainfall > 50: risk_score += 40
        elif rainfall > 25: risk_score += 20
        if soil_moisture > 70: risk_score += 30
        elif soil_moisture > 50: risk_score += 15
        if building_age > 50: risk_score += 20
        elif building_age > 30: risk_score += 10
        if temp > 35: risk_score += 10
        if wind > 20: risk_score += 10
        return 1 if risk_score > 50 else 0
    
    try:
        data = np.array(data).reshape(1, -1)
        result = model.predict(data)
        return int(result[0])
    except Exception as e:
        print(f"Prediction error: {e}")
        return 0  # Default to low risk on error