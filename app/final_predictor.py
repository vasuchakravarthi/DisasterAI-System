from predictor import predict_risk
from crack_predictor import predict_crack

def final_prediction(data, image_path):
    """
    data → environmental features
    image_path → uploaded image
    """

    # Step 1: Risk prediction
    risk = predict_risk(data)

    # Step 2: Crack detection
    crack_result = predict_crack(image_path)

    if "error" in crack_result:
        return crack_result

    damage_score = crack_result["damage_score"]
    status = crack_result["status"]

    # -----------------------------
    # FINAL DECISION LOGIC
    # -----------------------------
    if risk == 1 and status == "SEVERE DAMAGE":
        final_status = "CRITICAL ALERT 🚨"
    elif risk == 1 and status in ["MODERATE DAMAGE", "MINOR DAMAGE"]:
        final_status = "HIGH RISK ⚠️"
    elif risk == 0 and status == "SEVERE DAMAGE":
        final_status = "STRUCTURAL ALERT ⚠️"
    elif status == "MODERATE DAMAGE":
        final_status = "MODERATE WARNING"
    else:
        final_status = "SAFE / MONITOR"

    return {
        "risk_level": "HIGH" if risk == 1 else "LOW",
        "damage_score": damage_score,
        "structural_status": status,
        "final_status": final_status
    }