# app/crack_predictor.py
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_crack_model.h5")

_model = None

def load_crack_model():
    global _model
    if _model is None:
        print(f"📦 Loading model from: {MODEL_PATH}")
        if os.path.exists(MODEL_PATH):
            _model = load_model(MODEL_PATH)
            print(f"✅ Model loaded successfully")
        else:
            print(f"❌ Model not found at {MODEL_PATH}")
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return _model

def predict_crack(image_path, tile_size=256):
    try:
        model = load_crack_model()
        img = cv2.imread(image_path)
        if img is None:
            return {"error": True, "message": f"Cannot read image: {image_path}"}
        
        h, w = img.shape[:2]
        tiles = []
        
        for y in range(0, h - tile_size, tile_size):
            for x in range(0, w - tile_size, tile_size):
                tile = img[y:y+tile_size, x:x+tile_size]
                tiles.append(tile)
        
        if not tiles:
            return {"error": True, "message": "Image too small for analysis"}
        
        crack_tiles = 0
        for tile in tiles:
            tile_resized = cv2.resize(tile, (256, 256))
            tile_normalized = tile_resized.astype(np.float32) / 255.0
            tile_batch = np.expand_dims(tile_normalized, axis=0)
            prob = model.predict(tile_batch, verbose=0)[0][0]
            if prob > 0.5:
                crack_tiles += 1
        
        total_tiles = len(tiles)
        damage_score = (crack_tiles / total_tiles) * 100 if total_tiles > 0 else 0
        
        if damage_score < 5:
            status = "SAFE"
            severity = "LOW"
            recommendation = "No immediate action needed"
        elif damage_score < 20:
            status = "MINOR CRACKS"
            severity = "LOW"
            recommendation = "Monitor cracks, schedule inspection in 6 months"
        elif damage_score < 50:
            status = "MODERATE DAMAGE"
            severity = "MEDIUM"
            recommendation = "Schedule structural inspection within 30 days"
        else:
            status = "SEVERE DAMAGE"
            severity = "HIGH"
            recommendation = "IMMEDIATE ACTION REQUIRED"
        
        return {
            "error": False,
            "damage_score": round(damage_score, 2),
            "status": status,
            "severity": severity,
            "crack_tiles": crack_tiles,
            "total_tiles": total_tiles,
            "crack_probability": round(crack_tiles / total_tiles, 3),
            "confidence": round(crack_tiles / total_tiles, 3),
            "recommendation": recommendation,
            "crack_detected": crack_tiles > 0,
            "model_source": "best_crack_model.h5"
        }
    except Exception as e:
        return {"error": True, "message": str(e), "damage_score": 0, "status": "ERROR"}