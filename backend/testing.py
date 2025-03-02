import requests

API_KEY = "AIzaSyBg4TgD5IlgAf12dUh4S4U4B1NGJPeASBA"

def get_lat_long():
    url = "https://www.googleapis.com/geolocation/v1/geolocate?key=" + API_KEY
    response = requests.post(url, json={})
    data = response.json()
    # print(data)
    return {"latitude": data["location"]["lat"], "longitude": data["location"]["lng"]}

print(get_lat_long())
