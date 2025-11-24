import websocket
import json
import os
import time
import threading
from datetime import datetime
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

# Charger .env
load_dotenv()

# Connexion MongoDB
mongo_user = os.getenv("MONGO_USER")
mongo_password = os.getenv("MONGO_PASSWORD")
mongo_host = os.getenv("MONGO_HOST", "mongo")
mongo_port = int(os.getenv("MONGO_PORT", 27017))
mongo_db = os.getenv("MONGO_DB", "crypto")

client = MongoClient(f"mongodb://{mongo_user}:{mongo_password}@{mongo_host}:{mongo_port}/?authSource=admin")
db = client[mongo_db]
collection = db.klines

print(" MongoDB connecté !")

# Liste des symbols
SYMBOLS = ["btcusdt", "ethusdt", "bnbusdt", "solusdt"]

# Liste partagée pour stocker les données
shared_data = []
lock = threading.Lock()

# Fonction WebSocket callback
def on_message(ws, message):
    data = json.loads(message)
    try:
        k = data['k']
        doc = {
            "symbol": k['s'].lower(),
            "open": float(k['o']),
            "high": float(k['h']),
            "low": float(k['l']),
            "close": float(k['c']),
            "volume": float(k['v']),
            "open_time": datetime.fromtimestamp(k['t'] / 1000.0),
            "close_time": datetime.fromtimestamp(k['T'] / 1000.0),
            "closed": bool(k['x'])
        }
        with lock:
            shared_data.append(doc)
    except KeyError as e:
        print(f"KeyError: {e}")

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed")

def on_open(ws):
    print("WebSocket opened")

def start_websocket(symbol):
    ws_url = f"wss://stream.binance.com:9443/ws/{symbol}@kline_1m"
    ws = websocket.WebSocketApp(ws_url,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close,
                                on_open=on_open)
    ws.run_forever()

def process_data():
    while True:
        time.sleep(10)
        with lock:
            if shared_data:
                df = pd.DataFrame(shared_data)
                df_closed = df[df['closed'] == True]
                if not df_closed.empty:
                    collection.insert_many(df_closed.to_dict("records"))
                    print(f" Inséré {len(df_closed)} bougies fermées dans MongoDB")
                shared_data.clear()

# Lancer un thread pour chaque symbole
for symbol in SYMBOLS:
    threading.Thread(target=start_websocket, args=(symbol,), daemon=True).start()

# Thread pour traiter les données
threading.Thread(target=process_data, daemon=True).start()

# Boucle principale pour garder le script actif
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Arrêt du streaming...")
