import random
import requests
import time


API_URL = "http://127.0.0.1:8000/predict"

districts = [
    "Śródmieście",
    "Mokotów",
    "Wola",
    "Ochota",
    "Żoliborz",
    "Praga-Północ",
    "Praga-Południe",
    "Ursynów",
    "Bielany",
    "Bemowo",
    "Targówek",
    "Białołęka",
    "Wawer",
    "Wilanów",
    "Ursus"
]


for i in range(100):
    payload = {
        "distance_from_center_km": round(random.uniform(0.5, 18.0), 2),
        "area_m2": round(random.uniform(25, 120), 1),
        "floor": random.randint(0, 15),
        "district": random.choice(districts),
        "rent_pln": round(random.uniform(300, 2200), 2)
    }

    response = requests.post(API_URL, json=payload)

    print(i + 1, response.status_code, response.json())

    time.sleep(0.1)
