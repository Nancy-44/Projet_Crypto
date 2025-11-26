import os
import requests
from requests.auth import HTTPBasicAuth

# --------------------------
# VARIABLES D'ENVIRONNEMENT
# --------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")  
API_USER = os.getenv("API_USER")
API_PASSWORD = os.getenv("API_PASSWORD")

auth = HTTPBasicAuth(API_USER, API_PASSWORD)

# --------------------------
# FONCTIONS API
# --------------------------

def get_prediction(symbol: str):
    """
    POST /predict avec JSON {symbol}
    """
    url = f"{API_BASE_URL}/predict"
    try:
        response = requests.post(
            url,
            json={"symbol": symbol},
            auth=auth,
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def get_latest(symbol: str):
    """
    GET /latest?symbol=...
    """
    url = f"{API_BASE_URL}/latest"
    try:
        response = requests.get(
            url,
            params={"symbol": symbol},
            auth=auth,
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def get_historical(symbol: str, limit=200):
    """
    GET /historical?symbol=...&limit=...
    """
    url = f"{API_BASE_URL}/historical"
    try:
        response = requests.get(
            url,
            params={"symbol": symbol, "limit": limit},
            auth=auth,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}
