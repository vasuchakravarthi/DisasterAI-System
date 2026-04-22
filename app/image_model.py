# app/image_model.py

from app.crack_predictor import predict_crack

def analyze_building(path):
    """
    Uses NEW tile-based crack model
    """
    try:
        result = predict_crack(path)

        if "error" in result:
            return result

        return {
            "error": False,
            "condition": result["status"],
            "damage_score": result["damage_score"],
            "severity": result["status"],
            "crack_tiles": result["crack_tiles"],
            "total_tiles": result["total_tiles"]
        }

    except Exception as e:
        return {
            "error": True,
            "condition": "Analysis Failed",
            "damage_score": 0,
            "severity": "ERROR",
            "message": str(e)
        }


def analyze_building_safe(path):
    return analyze_building(path)