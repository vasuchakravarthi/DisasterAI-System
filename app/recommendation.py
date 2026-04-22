# recommendation.py - Already good, no changes needed
def recommend(risk):
    if risk == 1:
        return {
            "action": "Immediate repair required",
            "material": "Reinforced concrete",
            "time": "Within 7 days"
        }
    else:
        return {
            "action": "Regular monitoring",
            "material": "No change needed",
            "time": "Monthly check"
        }