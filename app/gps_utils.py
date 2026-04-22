"""
GPS Utils - Extract GPS coordinates from images
Location: app/gps_utils.py
"""

import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import re

def extract_gps_from_image(image_path):
    """
    Extract GPS coordinates from image EXIF data
    Returns: (latitude, longitude) or None
    """
    try:
        # Open image with PIL
        img = Image.open(image_path)
        exif_data = img._getexif()
        
        if not exif_data:
            return None
        
        # Get GPS info
        gps_info = {}
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == 'GPSInfo':
                for gps_tag_id, gps_value in value.items():
                    gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                    gps_info[gps_tag] = gps_value
        
        if not gps_info:
            return None
        
        # Convert GPS coordinates to decimal degrees
        def convert_to_degrees(value):
            d = float(value[0])
            m = float(value[1])
            s = float(value[2])
            return d + (m / 60.0) + (s / 3600.0)
        
        lat = convert_to_degrees(gps_info.get('GPSLatitude', [0,0,0]))
        lon = convert_to_degrees(gps_info.get('GPSLongitude', [0,0,0]))
        
        # Handle N/S and E/W
        if gps_info.get('GPSLatitudeRef') == 'S':
            lat = -lat
        if gps_info.get('GPSLongitudeRef') == 'W':
            lon = -lon
        
        return {
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "altitude": gps_info.get('GPSAltitude', 0)
        }
        
    except Exception as e:
        print(f"GPS extraction error: {e}")
        return None

def find_nearest_city(lat, lon, locations_db):
    """
    Find nearest major city to given coordinates
    """
    nearest = None
    min_distance = float('inf')
    
    for key, loc in locations_db.items():
        distance = ((loc["lat"] - lat) ** 2 + (loc["lon"] - lon) ** 2) ** 0.5
        if distance < min_distance:
            min_distance = distance
            nearest = key
    
    return nearest, min_distance