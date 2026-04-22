# structure.py - ENHANCED VERSION
import random

def get_structure_data():
    age = random.randint(5, 50)
    materials = ["Concrete", "Steel", "Brick", "Composite"]
    material = random.choice(materials)
    cracks = random.choice([0, 1])  # 0 = no, 1 = yes
    
    return {
        "age": age,
        "material": material,
        "cracks": "Yes" if cracks == 1 else "No",
        "damage_score": random.randint(5, 60) if cracks == 1 else random.randint(0, 20)
    }