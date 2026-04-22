# simulation.py - FIXED VERSION
from app.predictor import predict_risk

def future_prediction(data, increase_percent=30):
    new_data = data.copy()
    new_data[0] = new_data[0] * (1 + increase_percent/100)
    return predict_risk(new_data)

def simulate(data, rain_inc=20, temp_inc=2):
    new_data = data.copy()
    new_data[0] += rain_inc
    new_data[1] += temp_inc
    return predict_risk(new_data)