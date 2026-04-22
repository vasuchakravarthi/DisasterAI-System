"""
Multi-Location Support Module
Location: app/locations.py
"""

LOCATIONS = {
    "mumbai": {
        "name": "Mumbai",
        "lat": 19.0760,
        "lon": 72.8777,
        "description": "Coastal city, high humidity, clay soil",
        "risk_factors": ["High rainfall", "Coastal erosion", "Dense construction"],
        "soil_type": "Marine Clay",
        "avg_temperature": 27,
        "avg_rainfall": 220
    },
    "delhi": {
        "name": "Delhi",
        "lat": 28.6139,
        "lon": 77.2090,
        "description": "Urban center, extreme temperatures",
        "risk_factors": ["Temperature extremes", "Air pollution impact", "Old structures"],
        "soil_type": "Sandy Loam",
        "avg_temperature": 25,
        "avg_rainfall": 80
    },
    "chennai": {
        "name": "Chennai",
        "lat": 13.0827,
        "lon": 80.2707,
        "description": "Coastal, cyclone prone",
        "risk_factors": ["Cyclone risk", "Flooding", "High groundwater"],
        "soil_type": "Alluvial Clay",
        "avg_temperature": 28,
        "avg_rainfall": 140
    },
    "kolkata": {
        "name": "Kolkata",
        "lat": 22.5726,
        "lon": 88.3639,
        "description": "River delta, soft soil",
        "risk_factors": ["Soft soil", "High water table", "Cyclone risk"],
        "soil_type": "Alluvial",
        "avg_temperature": 26,
        "avg_rainfall": 180
    },
    "bangalore": {
        "name": "Bangalore",
        "lat": 12.9716,
        "lon": 77.5946,
        "description": "High altitude, rocky soil",
        "risk_factors": ["Rapid construction", "Groundwater depletion"],
        "soil_type": "Red Lateritic",
        "avg_temperature": 24,
        "avg_rainfall": 85
    },
    "hyderabad": {
        "name": "Hyderabad",
        "lat": 17.3850,
        "lon": 78.4867,
        "description": "Rocky terrain, moderate climate",
        "risk_factors": ["Rocky soil challenges", "Expanding urban area"],
        "soil_type": "Red Sandy",
        "avg_temperature": 26,
        "avg_rainfall": 75
    },
    "ahmedabad": {
        "name": "Ahmedabad",
        "lat": 23.0225,
        "lon": 72.5714,
        "description": "Arid climate, industrial hub",
        "risk_factors": ["Water scarcity", "Heat waves"],
        "soil_type": "Sandy Desert",
        "avg_temperature": 28,
        "avg_rainfall": 50
    },
    "pune": {
        "name": "Pune",
        "lat": 18.5204,
        "lon": 73.8567,
        "description": "Plateau city, moderate climate",
        "risk_factors": ["Urban sprawl", "Hill slope issues"],
        "soil_type": "Basaltic",
        "avg_temperature": 25,
        "avg_rainfall": 95
    }
}

def get_location_info(location_key):
    """Get location information by key"""
    return LOCATIONS.get(location_key, LOCATIONS["mumbai"])

def get_all_locations():
    """Get all available locations"""
    return list(LOCATIONS.keys())

def get_location_coordinates(location_key):
    """Get coordinates for a location"""
    loc = LOCATIONS.get(location_key, LOCATIONS["mumbai"])
    return loc["lat"], loc["lon"]

def get_location_risk_factors(location_key):
    """Get risk factors for a location"""
    loc = LOCATIONS.get(location_key, LOCATIONS["mumbai"])
    return loc.get("risk_factors", [])