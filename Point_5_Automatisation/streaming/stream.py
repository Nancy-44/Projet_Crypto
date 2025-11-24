
import os
import pandas as pd
from pymongo import MongoClient
from datetime import datetime, timedelta

# ----------------------------
# Variables d'environnement Mongo
# ----------------------------
MONGO_HOST = os.getenv("MONGO_HOST")
MONGO_PORT = int(os.getenv("MONGO_PORT"))
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
MONGO_DB = os.getenv("MONGO_DB")

# ----------------------------
# Connexion MongoDB
# ----------------------------
client = MongoClient(
    f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/?authSource=admin"
)
db = client[MONGO_DB]
collection = db.klines

print(f"Connecté à MongoDB {MONGO_HOST}:{MONGO_PORT}, DB: {MONGO_DB}")

# ----------------------------
# Paramètres
# ----------------------------
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
INTERVALS = {
    "15m": 15,
    "1h": 60,
    "4h": 240,
    "1d": 1440
}
NUM_CANDLES = 60  # nombre de bougies à insérer

# ----------------------------
# Insertion des bougies
# ----------------------------
for symbol in SYMBOLS:
    now = datetime.now()
    for interval_name, interval_min in INTERVALS.items():
        for i in range(NUM_CANDLES):
            close_time = now - timedelta(minutes=i*interval_min)
            open_time = close_time - timedelta(minutes=interval_min)
            doc = {
                "symbol": symbol,
                "interval": interval_name,  
                "open_time": open_time,
                "close_time": close_time,
                "open": 100 + i*0.1,
                "high": 100 + i*0.2,
                "low": 100 + i*0.05,
                "close": 100 + i*0.15,
                "volume": 10 + i*0.5,
                "closed": True
            }
            collection.insert_one(doc)
        print(f"{NUM_CANDLES} bougies {interval_name} insérées pour {symbol}")

print("Insertion terminée")
