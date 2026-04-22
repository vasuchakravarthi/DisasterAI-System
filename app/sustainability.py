# sustainability.py - Already good, no changes needed
def sustainability(risk):
    if risk == 1:
        return {
            "cost": "High",
            "carbon": "Medium",
            "durability": "10 years"
        }
    else:
        return {
            "cost": "Low",
            "carbon": "Low",
            "durability": "20 years"
        }