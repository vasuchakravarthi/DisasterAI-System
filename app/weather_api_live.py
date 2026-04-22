# weather_api_live.py - REAL-TIME WEATHER
import requests

def get_live_weather():
    """Get real-time weather from Open-Meteo API"""
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=16.54&longitude=81.52&current_weather=true&hourly=relative_humidity_2m,precipitation&timezone=auto"
        
        response = requests.get(url, timeout=10)
        data = response.json()
        
        # Current weather
        current = data.get("current_weather", {})
        temp = current.get("temperature", 28)
        wind = current.get("windspeed", 10)
        
        # Get precipitation from hourly data
        hourly = data.get("hourly", {})
        precip = hourly.get("precipitation", [0])[0] if hourly else 0
        
        # Get humidity
        humidity = hourly.get("relative_humidity_2m", [65])[0] if hourly else 65
        
        return (float(precip), float(temp), float(wind), float(humidity))
        
    except Exception as e:
        print(f"Live weather error: {e}")
        return None