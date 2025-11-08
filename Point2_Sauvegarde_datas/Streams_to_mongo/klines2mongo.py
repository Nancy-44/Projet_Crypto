from binance.client import Client
from pymongo import MongoClient
import datetime
import time

binance_api_key = 'YOUR_BINANCE_API_KEY'
binance_api_secret = 'YOUR_BINANCE_API_SECRET'

symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']  
interval = Client.KLINE_INTERVAL_1HOUR       
limit = 500  

binance_client = Client(api_key=binance_api_key, api_secret=binance_api_secret)
mongo_client = MongoClient(
    host = "127.0.0.1",
    port = 27017,
    username = "datascientest", # TODO use env
    password = "dst123"         # TODO use env
)
db = mongo_client['binance_klines']  # Nom de la base

for symbol in symbols:
    print(f"Récupération des klines pour {symbol}...")

    try:
        klines = binance_client.get_klines(symbol=symbol, interval=interval, limit=limit)

        formatted_klines = []
        for k in klines:
            kline = {
                'open_time': datetime.datetime.fromtimestamp(k[0] / 1000),
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5]),
                'close_time': datetime.datetime.fromtimestamp(k[6] / 1000),
                'symbol': symbol
            }
            formatted_klines.append(kline)

        # Insertion dans MongoDB
        collection = db[symbol]
        result = collection.insert_many(formatted_klines)
        print(f"SUCCESS {len(result.inserted_ids)} klines insérés pour {symbol}.")

    except Exception as e:
        print(f"Erreur pour {symbol}: {e}")
        time.sleep(1)  # Pause pour éviter le ban API