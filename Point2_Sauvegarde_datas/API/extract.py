## Extraction des données OHLCV depuis l’API Binance et sauvegarde des données bruts en JSON.
## Les données sont extraites pour chaque symbole et intervalle défini dans la configuration (boucle jusqu'à 10000 points par symbol mais filtrer les intervalles et symbol pour ne pas planter API).

import requests
import json
import os
import time
from config import BASE_URL, KLINE_ENDPOINT, SYMBOLS, INTERVALS

def fetch_klines(symbol, interval, limit=1000, end_time=None):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_time:
        params["endTime"] = end_time
    url = f"{BASE_URL}{KLINE_ENDPOINT}"
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()

def fetch_historical_data(symbol, interval, total_points=10000):
    all_data = []
    end_time = None

    while len(all_data) < total_points:
        data = fetch_klines(symbol, interval, 1000, end_time)
        if not data:
            break
        all_data.extend(data)
        # Préparer le prochain batch (aller plus loin dans le passé)
        end_time = data[0][0]  # open_time du premier kline
        time.sleep(0.1)  # éviter rate limit API

    print(f"{symbol} - {interval} : {len(all_data)} points récupérés.")
    return all_data

def save_raw_data(symbol, interval, data):
    os.makedirs("data/raw", exist_ok=True)
    filename = f"data/raw/{symbol}_{interval}.json"
    with open(filename, "w") as f:
        json.dump(data, f)
    print(f"Données enregistrées : {filename}")

def run_extraction():
    for symbol in SYMBOLS:
        for interval in INTERVALS:
            data = fetch_historical_data(symbol, interval, total_points=10000)
            save_raw_data(symbol, interval, data)

if __name__ == "__main__":
    run_extraction()