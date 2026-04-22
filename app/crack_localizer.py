"""
Crack Localizer - Extract and refine bounding boxes from Grad-CAM heatmaps
Location: app/crack_localizer.py
"""

import cv2
import numpy as np

def extract_bounding_boxes(heatmap, threshold=0.3, min_area=100):
    """
    Extract bounding boxes from heatmap using multiple threshold strategies
    """
    # Try multiple thresholds for better detection
    thresholds = [0.25, 0.3, 0.35, 0.4]
    all_boxes = []
    
    for thresh in thresholds:
        _, binary = cv2.threshold(np.uint8(255 * heatmap), int(thresh * 255), 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                box_region = heatmap[y:y+h, x:x+w]
                confidence = float(np.mean(box_region))
                
                all_boxes.append({
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                    "area": int(area),
                    "confidence": round(confidence, 3),
                    "aspect_ratio": round(w / h, 2) if h > 0 else 0
                })
    
    # Remove overlapping boxes (Non-Maximum Suppression)
    unique_boxes = non_max_suppression(all_boxes, iou_threshold=0.5)
    
    # Sort by confidence
    unique_boxes.sort(key=lambda x: x['confidence'], reverse=True)
    
    return unique_boxes[:5]

def non_max_suppression(boxes, iou_threshold=0.5):
    """
    Remove overlapping bounding boxes
    """
    if len(boxes) == 0:
        return []
    
    boxes = sorted(boxes, key=lambda x: x['confidence'], reverse=True)
    result = []
    
    while boxes:
        current = boxes.pop(0)
        result.append(current)
        
        boxes = [box for box in boxes if calculate_iou(current, box) < iou_threshold]
    
    return result

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union between two boxes
    """
    x1 = max(box1['x'], box2['x'])
    y1 = max(box1['y'], box2['y'])
    x2 = min(box1['x'] + box1['width'], box2['x'] + box2['width'])
    y2 = min(box1['y'] + box1['height'], box2['y'] + box2['height'])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = box1['width'] * box1['height']
    area2 = box2['width'] * box2['height']
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def classify_crack_type(box):
    """
    Classify crack type based on bounding box aspect ratio
    """
    aspect_ratio = box['aspect_ratio']
    
    if aspect_ratio > 3:
        return "Horizontal Crack"
    elif aspect_ratio < 0.33:
        return "Vertical Crack"
    elif 0.8 < aspect_ratio < 1.2 and box['width'] > 30:
        return "Pothole / Spalling"
    else:
        return "Diagonal / Network Crack"

def draw_bounding_boxes(image, boxes):
    """
    Draw bounding boxes on image with labels
    """
    img_copy = image.copy()
    
    for i, box in enumerate(boxes):
        x, y, w, h = box['x'], box['y'], box['width'], box['height']
        confidence = box['confidence']
        crack_type = classify_crack_type(box)
        
        # Color based on confidence
        if confidence > 0.7:
            color = (0, 0, 255)  # Red - High confidence
        elif confidence > 0.4:
            color = (0, 165, 255)  # Orange - Medium confidence
        else:
            color = (0, 255, 255)  # Yellow - Low confidence
        
        # Draw rectangle
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)
        
        # Draw label
        label = f"{crack_type} ({confidence:.0%})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(img_copy, (x, y - label_size[1] - 4), (x + label_size[0], y), color, -1)
        cv2.putText(img_copy, label, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img_copy