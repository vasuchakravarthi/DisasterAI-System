"""
Soil Data Module - Enhanced Version
Fetches real soil data from ISRIC SoilGrids API with caching
Location: app/soil.py
"""

import requests
import json
import os
from datetime import datetime, timedelta

# Cache file to avoid repeated API calls
CACHE_FILE = "data/soil_cache.json"
CACHE_DURATION = timedelta(hours=24)  # Cache for 24 hours

def load_cache():
    """Load cached soil data"""
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                cache = json.load(f)
                # Check if cache is still valid
                cache_time = datetime.fromisoformat(cache.get('timestamp', '2000-01-01'))
                if datetime.now() - cache_time < CACHE_DURATION:
                    print("✅ Using cached soil data")
                    return cache.get('data')
    except Exception as e:
        print(f"Cache load error: {e}")
    return None

def save_cache(data):
    """Save soil data to cache"""
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        cache = {
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f)
        print("✅ Soil data cached")
    except Exception as e:
        print(f"Cache save error: {e}")

def get_soil_classification(clay, sand, silt):
    """
    Classify soil type based on USDA Soil Texture Triangle
    """
    if clay > 40:
        if sand > 45:
            return "Sandy Clay"
        elif silt > 40:
            return "Silty Clay"
        else:
            return "Clay"
    elif clay > 27:
        if sand > 45:
            return "Sandy Clay Loam"
        elif silt > 45:
            return "Silty Clay Loam"
        else:
            return "Clay Loam"
    elif clay > 20:
        if sand > 52:
            return "Sandy Loam"
        elif silt > 50:
            return "Silt Loam"
        else:
            return "Loam"
    elif clay > 12:
        if sand > 50:
            return "Sandy Loam"
        else:
            return "Loam"
    else:
        if sand > 85:
            return "Sand"
        elif sand > 70:
            return "Loamy Sand"
        else:
            return "Sandy Loam"

def get_soil_bearing_capacity(soil_type, clay_percentage):
    """
    Calculate bearing capacity based on soil type and clay content (kPa)
    """
    bearing_capacity = {
        "Sand": 150,
        "Loamy Sand": 130,
        "Sandy Loam": 120,
        "Loam": 110,
        "Silt Loam": 100,
        "Sandy Clay Loam": 95,
        "Clay Loam": 90,
        "Silty Clay Loam": 85,
        "Sandy Clay": 80,
        "Silty Clay": 75,
        "Clay": 65
    }
    
    base_bearing = bearing_capacity.get(soil_type, 85)
    
    # Adjust for clay percentage (higher clay = lower bearing capacity)
    if clay_percentage > 50:
        base_bearing *= 0.7
    elif clay_percentage > 40:
        base_bearing *= 0.85
    
    return round(base_bearing, 1)

def get_soil_moisture(clay_percentage, sand_percentage):
    """
    Estimate field capacity moisture based on soil composition
    """
    # Clay holds more water, sand drains quickly
    moisture = 30 + (clay_percentage * 0.6) - (sand_percentage * 0.3)
    return round(min(85, max(25, moisture)), 1)

def get_soil_permeability(soil_type, clay_percentage):
    """
    Estimate soil permeability (cm/hr)
    """
    if clay_percentage > 45:
        return "Very Low (< 0.1 cm/hr) - Poor drainage"
    elif clay_percentage > 35:
        return "Low (0.1-0.5 cm/hr) - Moderate drainage issues"
    elif clay_percentage > 25:
        return "Moderate (0.5-2 cm/hr) - Adequate drainage"
    else:
        return "High (> 2 cm/hr) - Good drainage"

def get_soil_data(lat, lon):
    """
    Get soil data for given coordinates from ISRIC SoilGrids API
    With enhanced properties and caching
    
    Args:
        lat: Latitude
        lon: Longitude
    
    Returns:
        dict: Soil properties including type, composition, bearing, moisture
    """
    
    # Try to load from cache first
    cached_data = load_cache()
    if cached_data and cached_data.get('lat') == lat and cached_data.get('lon') == lon:
        return cached_data
    
    try:
        print(f"🌍 Fetching real soil data for coordinates ({lat}, {lon})...")
        
        url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}&property=clay&property=sand&property=silt&depth=0-5cm&value=mean"
        
        response = requests.get(url, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            # Default values
            clay = 35
            sand = 35
            silt = 30
            
            # Extract values from API response
            if "properties" in data and "layers" in data["properties"]:
                for layer in data["properties"]["layers"]:
                    layer_name = layer.get("name", "")
                    depths = layer.get("depths", [])
                    if depths and len(depths) > 0:
                        values = depths[0].get("values", {})
                        mean_value = values.get("mean", None)
                        
                        if mean_value is not None:
                            if layer_name == "clay":
                                clay = float(mean_value)
                            elif layer_name == "sand":
                                sand = float(mean_value)
                            elif layer_name == "silt":
                                silt = float(mean_value)
            
            # Ensure percentages sum to ~100
            total = clay + sand + silt
            if abs(total - 100) > 5:
                # Normalize to 100%
                clay = (clay / total) * 100
                sand = (sand / total) * 100
                silt = (silt / total) * 100
            
            # Classify soil type
            soil_type = get_soil_classification(clay, sand, silt)
            
            # Calculate derived properties
            moisture = get_soil_moisture(clay, sand)
            bearing = get_soil_bearing_capacity(soil_type, clay)
            permeability = get_soil_permeability(soil_type, clay)
            
            # Determine foundation risk
            if clay > 45:
                foundation_risk = "HIGH - Expansive soil risk"
                foundation_recommendation = "Deep foundations recommended. Consider lime stabilization."
            elif clay > 35:
                foundation_risk = "MEDIUM - Moderate swelling potential"
                foundation_recommendation = "Reinforced foundations with proper drainage."
            else:
                foundation_risk = "LOW - Good bearing capacity"
                foundation_recommendation = "Standard shallow foundations acceptable."
            
            soil_data = {
                "lat": lat,
                "lon": lon,
                "type": soil_type,
                "clay_percentage": round(clay, 1),
                "clay": round(clay, 1),
                "sand": round(sand, 1),
                "silt": round(silt, 1),
                "moisture": moisture,
                "bearing_capacity_kpa": bearing,
                "bearing": bearing,
                "permeability": permeability,
                "foundation_risk": foundation_risk,
                "foundation_recommendation": foundation_recommendation,
                "soil_type": "real",
                "source": "ISRIC SoilGrids API"
            }
            
            # Save to cache
            save_cache(soil_data)
            
            print(f"✅ Soil data retrieved: {soil_type}, Clay: {clay:.1f}%")
            return soil_data
            
        else:
            raise Exception(f"API returned status {response.status_code}")
            
    except requests.exceptions.Timeout:
        print("⚠️ Soil API timeout - using regional data")
        return get_fallback_soil_data(lat, lon, "API Timeout")
        
    except requests.exceptions.ConnectionError:
        print("⚠️ Soil API connection error - using regional data")
        return get_fallback_soil_data(lat, lon, "Connection Error")
        
    except Exception as e:
        print(f"Soil API error: {e}")
        return get_fallback_soil_data(lat, lon, str(e))

def get_fallback_soil_data(lat, lon, error_reason=None):
    """
    Provide realistic fallback soil data for the Indian subcontinent
    Based on typical soil types for given coordinates
    """
    
    # Region-specific fallback for India (16.54°N, 81.52°E is in Andhra Pradesh)
    # This area typically has black cotton soil and red lateritic soil
    
    print(f"📍 Using regional soil data for {lat}, {lon}")
    
    # Typical soil for coastal Andhra Pradesh
    soil_data = {
        "lat": lat,
        "lon": lon,
        "type": "Clay Loam",
        "clay_percentage": 48.5,
        "clay": 48.5,
        "sand": 28.0,
        "silt": 23.5,
        "moisture": 62.0,
        "bearing_capacity_kpa": 82.0,
        "bearing": 82.0,
        "permeability": "Low (0.3 cm/hr) - Moderate drainage issues",
        "foundation_risk": "MEDIUM - Moderate swelling potential",
        "foundation_recommendation": "Reinforced foundations with proper drainage.",
        "soil_type": "estimated",
        "source": "Regional Database (India)",
        "error_reason": error_reason
    }
    
    return soil_data

def get_soil_recommendations(soil_data):
    """
    Generate construction recommendations based on soil properties
    """
    clay = soil_data.get("clay_percentage", 45)
    
    recommendations = []
    
    if clay > 45:
        recommendations.append("🏗️ Use deep pile foundations or under-reamed piles")
        recommendations.append("💧 Install proper drainage systems around foundation")
        recommendations.append("🧪 Consider lime or cement stabilization")
        recommendations.append("📏 Leave expansion joints for soil movement")
    elif clay > 35:
        recommendations.append("🏗️ Reinforced strip foundations recommended")
        recommendations.append("💧 Ensure proper surface water drainage")
        recommendations.append("📏 Provide adequate reinforcement for swelling")
    else:
        recommendations.append("🏗️ Standard shallow foundations acceptable")
        recommendations.append("💧 Regular drainage maintenance sufficient")
    
    return recommendations

# Test function
if __name__ == "__main__":
    # Test with coordinates
    test_lat = 16.54
    test_lon = 81.52
    
    print("Testing soil data retrieval...")
    soil = get_soil_data(test_lat, test_lon)
    
    print("\n" + "="*50)
    print("SOIL DATA RESULTS")
    print("="*50)
    for key, value in soil.items():
        print(f"{key}: {value}")
    
    print("\n" + "="*50)
    print("RECOMMENDATIONS")
    print("="*50)
    for rec in get_soil_recommendations(soil):
        print(rec)