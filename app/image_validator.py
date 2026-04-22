"""
Image Validator - FINAL FIXED VERSION
Balanced Validation System

Rejects:
- People
- Cars (only cars)
- Pure text documents
- Nature scenes

Warns (but allows):
- Medical-like grayscale images
- Text overlays

Location: app/image_validator.py
"""

import cv2
import numpy as np
import os

class ImageValidator:
    def __init__(self):
        cascade_face_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_face_path)
        self.car_cascade = None

        cascade_car_path = cv2.data.haarcascades + 'haarcascade_car.xml'
        if os.path.exists(cascade_car_path):
            self.car_cascade = cv2.CascadeClassifier(cascade_car_path)

    # -----------------------------------------------------
    # 🔥 FIXED MEDICAL DETECTION (NO REJECTION)
    # -----------------------------------------------------
    def is_medical_xray(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / (hist.sum() + 1e-8)

        hist_variance = np.var(hist)
        hist_peak = np.max(hist)

        h, w = gray.shape
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        center_brightness = np.mean(center_region)
        overall_brightness = np.mean(gray)

        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)

        has_curved_lines = False
        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 != 0:
                    angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    angles.append(angle)
            if len(angles) > 15 and np.std(angles) > 35:
                has_curved_lines = True

        # 🔥 RELAXED thresholds (important)
        if (hist_peak > 0.12 and hist_variance < 0.0002) or \
           (center_brightness < overall_brightness - 45) or \
           (has_curved_lines and lines is not None and len(lines) > 40):
            return True

        return False

    # -----------------------------------------------------
    # TEXT DETECTION (WARNING ONLY)
    # -----------------------------------------------------
    def has_text_content(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        horizontal_density = np.sum(horizontal_lines > 0) / (horizontal_lines.size + 1e-8)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        small_contours = sum(1 for c in contours if 50 < cv2.contourArea(c) < 800)

        std_dev = np.std(gray)

        if horizontal_density > 0.03 and small_contours > 20:
            return True
        if small_contours > 30 and std_dev < 70:
            return True

        return False

    # -----------------------------------------------------
    # PURE TEXT DOCUMENT (REJECT)
    # -----------------------------------------------------
    def is_pure_text_document(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.size + 1e-8)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)

        if lines is not None and len(lines) > 8 and edge_density > 0.05:
            return False

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        small_contours = sum(1 for c in contours if 50 < cv2.contourArea(c) < 800)

        if small_contours > 60:
            return True

        return False

    # -----------------------------------------------------
    # CAR DETECTION
    # -----------------------------------------------------
    def is_car_image(self, img, gray):
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.size + 1e-8)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
        has_structural = lines is not None and len(lines) > 8 and edge_density > 0.05

        if has_structural:
            return False, 0

        if self.car_cascade is not None:
            cars = self.car_cascade.detectMultiScale(gray, 1.1, 2)
            if len(cars) > 2:
                return True, len(cars)

        return False, 0

    # -----------------------------------------------------
    # STRUCTURE DETECTION
    # -----------------------------------------------------
    def is_structural_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.size + 1e-8)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)

        if edge_density > 0.04 and lines is not None:
            return True

        return False

    # -----------------------------------------------------
    # MAIN VALIDATION
    # -----------------------------------------------------
    def validate_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return False, "Could not read image", {}, None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        warning = None

        # 1. Face
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            return False, "❌ PERSON DETECTED!", {}, None

        # 2. Cars
        is_car, _ = self.is_car_image(img, gray)
        if is_car:
            return False, "❌ CAR IMAGE DETECTED!", {}, None

        # 3. 🔥 MEDICAL → WARNING ONLY
        if self.is_medical_xray(img):
            warning = "⚠️ Image looks grayscale/low texture. Proceeding with caution."

        # 4. Text document
        if self.is_pure_text_document(img):
            return False, "❌ TEXT DOCUMENT DETECTED!", {}, None

        # 5. Text overlay warning
        if self.has_text_content(img):
            warning = "⚠️ Text detected in image."

        # 6. Structural check
        if not self.is_structural_image(img):
            return False, "❌ NOT A STRUCTURAL IMAGE!", {}, None

        return True, "✅ Valid structural image detected", {}, warning


# Singleton
validator = ImageValidator()

def validate_structural_image(image_path):
    return validator.validate_image(image_path)