# weather_api.py - Already good, no changes needed
import requests

def get_nasa_data():
    try:
        url = "https://power.larc.nasa.gov/api/temporal/daily/point?parameters=PRECTOTCORR,T2M,WS2M,RH2M&community=AG&longitude=81.52&latitude=16.54&start=20220101&end=20221231&format=JSON"
        
        response = requests.get(url, timeout=15)
        
        if not response.text:
            print("❌ NASA EMPTY RESPONSE")
            return None
        
        try:
            data = response.json()
        except:
            print("❌ NASA NOT JSON")
            return None
        
        params = data.get("properties", {}).get("parameter", {})
        
        rainfall = list(params.get("PRECTOTCORR", {}).values())
        temperature = list(params.get("T2M", {}).values())
        wind = list(params.get("WS2M", {}).values())
        humidity = list(params.get("RH2M", {}).values())
        
        if not rainfall:
            print("❌ NASA NO DATA")
            return None
        
        return rainfall, temperature, wind, humidity
        
    except Exception as e:
        print("❌ NASA ERROR:", e)
        return None