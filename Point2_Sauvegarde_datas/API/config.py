## Centralise la configuration : URL, API, bases de données, symboles, intervalles.
# Avec fichier extract.py filtrer 15m, 1 et 4 d'abord (10000 point total), puis 1d (500 points total) et 1w (200 points total) après.

from sqlalchemy import create_engine

# Base PostgreSQL (le hostname = nom du service docker)
DATABASE_URL = "postgresql://ubuntu:postgres@postgres:5432/crypto_db"
engine = create_engine(DATABASE_URL)

# Binance API
BASE_URL = "https://api.binance.com"
KLINE_ENDPOINT = "/api/v3/klines"

# Liste des symboles et intervalles à extraire
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
INTERVALS = ["15m", "1h", "4h", "1d", "1w"]

INTERVAL_SECONDS = {
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
   "1w": 604800
}

LIMIT = 1000  # max autorisé par Binance