"""
SDNET2018 Crack Detection - Using Your Colab Grad-CAM
Location: app/sdnet2018_predictor.py
"""

import tensorflow as tf
import numpy as np
import cv2
import os
import base64
from app.gradcam import generate_crack_visualization

class SDNET2018Predictor:
    def __init__(self, model_path="models/crack_model.keras"):
        self.model_path = model_path
        self.model = None
        self.threshold = 0.3
        self.load_model()
    
    def load_model(self):
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                print(f"✅ Model loaded from {self.model_path}")
            else:
                alt_path = "models/crack_detection_model.h5"
                if os.path.exists(alt_path):
                    self.model = tf.keras.models.load_model(alt_path)
                    print(f"✅ Model loaded from {alt_path}")
                else:
                    print(f"⚠️ Model not found")
                    self.model = None
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def generate_visualizations(self, image_path):
        """
        Generate all visualizations using YOUR Colab Grad-CAM code
        """
        if self.model is None:
            return self._error_response("Model not loaded")
        
        try:
            # Generate visualization using your Colab function
            result = generate_crack_visualization(self.model, image_path, self.threshold)
            
            # Convert images to base64 for frontend
            overlay_base64 = self._img_to_base64(result["overlay"])
            heatmap_color_base64 = self._img_to_base64(result["heatmap_color"])
            heatmap_alone_base64 = self._img_to_base64(result["heatmap_alone"])
            
            damage_score = result["crack_probability"] * 100
            
            return {
                "success": True,
                "crack_probability": result["crack_probability"],
                "damage_score": round(damage_score, 2),
                "crack_detected": result["crack_detected"],
                "confidence": result["confidence"],
                "crack_count": len(result["bounding_boxes"]),
                "bounding_boxes": result["bounding_boxes"],
                "image_with_boxes": overlay_base64,
                "heatmap_overlay": heatmap_color_base64,
                "heatmap_alone": heatmap_alone_base64
            }
            
        except Exception as e:
            print(f"❌ Visualization error: {e}")
            import traceback
            traceback.print_exc()
            return self._error_response(str(e))
    
    def _img_to_base64(self, img):
        """Convert image to base64 string"""
        _, buffer = cv2.imencode('.png', img)
        return base64.b64encode(buffer).decode('utf-8')
    
    def _error_response(self, message):
        return {
            "success": False,
            "error": message,
            "crack_probability": 0,
            "damage_score": 0,
            "crack_detected": False,
            "confidence": 0,
            "crack_count": 0,
            "bounding_boxes": [],
            "image_with_boxes": None,
            "heatmap_overlay": None,
            "heatmap_alone": None
        }
    
    def predict(self, image_path):
        """Simple prediction without visualization"""
        if self.model is None:
            return 0.5, 0.5, False
        
        img = cv2.imread(image_path)
        img_resized = cv2.resize(img, (224, 224)) / 255.0
        img_array = np.expand_dims(img_resized, axis=0)
        pred = self.model.predict(img_array, verbose=0)[0][0]
        crack_detected = pred > self.threshold
        confidence = pred if crack_detected else 1 - pred
        
        return float(pred), float(confidence), bool(crack_detected)
    
    def get_damage_score(self, image_path):
        pred, _, _ = self.predict(image_path)
        return round(pred * 100, 2)
    
    def analyze(self, image_path):
        result = self.generate_visualizations(image_path)
        
        if result.get("crack_probability", 0) > 0.7:
            severity = "HIGH"
            condition = "Severe Crack Detected"
            recommendation = "Immediate structural inspection required"
        elif result.get("crack_probability", 0) > 0.4:
            severity = "MEDIUM"
            condition = "Moderate Crack Detected"
            recommendation = "Schedule inspection within 1 month"
        elif result.get("crack_probability", 0) > 0.2:
            severity = "LOW"
            condition = "Minor Crack Detected"
            recommendation = "Monitor regularly"
        else:
            severity = "LOW"
            condition = "No Significant Cracks"
            recommendation = "Routine maintenance only"
        
        return {
            "crack_probability": result.get("crack_probability", 0),
            "damage_score": result.get("damage_score", 0),
            "confidence": result.get("confidence", 0.5),
            "crack_detected": result.get("crack_detected", False),
            "severity": severity,
            "condition": condition,
            "recommendation": recommendation,
            "crack_count": result.get("crack_count", 0),
            "bounding_boxes": result.get("bounding_boxes", []),
            "image_with_boxes": result.get("image_with_boxes"),
            "heatmap_overlay": result.get("heatmap_overlay"),
            "heatmap_alone": result.get("heatmap_alone"),
            "model": "SDNET2018 CNN + Grad-CAM"
        }

# Global instance
predictor = SDNET2018Predictor()

def get_damage_score(image_path):
    return predictor.get_damage_score(image_path)

def analyze_structural_image(image_path):
    return predictor.analyze(image_path)

def generate_visualizations(image_path):
    return predictor.generate_visualizations(image_path)