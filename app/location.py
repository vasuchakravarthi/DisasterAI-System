import requests

def get_location_data(lat, lon):
    try:
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
        res = requests.get(url).json()

        elevation = res["results"][0]["elevation"]

        region = "Flood-prone" if elevation < 20 else "Safe zone"

        return {
            "latitude": lat,
            "longitude": lon,
            "elevation": elevation,
            "region": region
        }

    except:
        return {
            "latitude": lat,
            "longitude": lon,
            "elevation": 10,
            "region": "Unknown"
        }