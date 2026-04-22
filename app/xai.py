# xai.py - Explainable AI Module
def explain(features):
    """
    Explain why risk is high or low
    features: [rainfall, temp, humidity, wind, soil_moisture, building_age]
    """
    rainfall, temp, humidity, wind, soil_moisture, building_age = features
    
    reasons = []
    
    if rainfall > 50:
        reasons.append(f"Heavy rainfall ({rainfall}mm) - High flood risk")
    elif rainfall > 25:
        reasons.append(f"Moderate rainfall ({rainfall}mm) - Monitor conditions")
    else:
        reasons.append(f"Low rainfall ({rainfall}mm) - Favorable")
    
    if soil_moisture > 70:
        reasons.append(f"High soil moisture ({soil_moisture}%) - Foundation risk")
    elif soil_moisture > 50:
        reasons.append(f"Elevated soil moisture - Monitor settlement")
    
    if building_age > 50:
        reasons.append(f"Building age ({building_age} years) - Structural concerns")
    elif building_age > 30:
        reasons.append(f"Building age ({building_age} years) - Increased maintenance")
    
    if wind > 20:
        reasons.append(f"Strong winds ({wind} m/s) - Lateral loading")
    
    if len(reasons) == 0:
        reasons.append("All parameters within normal ranges")
    
    return {
        "reasons": reasons,
        "summary": reasons[0] if reasons else "Normal conditions"
    }