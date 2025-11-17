import requests
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.types import Float, DateTime, Integer
import os
import time
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv

# --- Charger les variables d'environnement ---
load_dotenv("/app/.env")  # chemin vers .env dans docker

# --- Connexion PostgreSQL ---
DB_USER = os.getenv("POSTGRES_USER")
DB_PASS = os.getenv("POSTGRES_PASSWORD")
DB_NAME = os.getenv("POSTGRES_DB")
DB_HOST = os.getenv("POSTGRES_HOST")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# --- Paramètres API Binance ---
BASE_URL = "https://api.binance.com"
KLINE_ENDPOINT = "/api/v3/klines"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
INTERVALS = ["1m", "5m", "15m", "1h", "4h", "1d"]
LIMIT = 20
INTERVAL_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}

# --- Fonctions utilitaires ---
def wait_for_postgres(max_retries=10, delay=3):
    retries = 0
    while retries < max_retries:
        try:
            with engine.connect() as conn:
                print("PostgreSQL est prêt !")
                return True
        except OperationalError:
            retries += 1
            print(f"PostgreSQL pas encore prêt, retry {retries}/{max_retries}...")
            time.sleep(delay)
    raise Exception("Impossible de se connecter à PostgreSQL après plusieurs tentatives")

def is_klines_empty():
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM klines"))
            count = result.scalar()
            return count == 0
    except Exception:
        return True  # table n'existe pas encore

# --- Création des tables ---
def create_tables():
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS symbol (
                symbol_id SERIAL PRIMARY KEY,
                symbol VARCHAR(10) UNIQUE NOT NULL,
                base_asset VARCHAR(10),
                quote_asset VARCHAR(10)
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS intervals (
                interval_id SERIAL PRIMARY KEY,
                interval_name VARCHAR(10) UNIQUE NOT NULL,
                seconds INTEGER NOT NULL
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS klines (
                klines_id SERIAL PRIMARY KEY,
                symbol_id INT REFERENCES symbol(symbol_id),
                interval_id INT REFERENCES intervals(interval_id),
                open_time TIMESTAMP,
                open FLOAT,
                high FLOAT,
                low FLOAT,
                close FLOAT,
                volume FLOAT,
                close_time TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))

# --- Fonctions auxiliaires ---
def get_or_create_symbol(symbol):
    base_asset = symbol[:-4]
    quote_asset = symbol[-4:]
    with engine.begin() as conn:
        result = conn.execute(text("SELECT symbol_id FROM symbol WHERE symbol = :symbol"), {"symbol": symbol}).fetchone()
        if result:
            return result[0]
        conn.execute(text(
            "INSERT INTO symbol (symbol, base_asset, quote_asset) VALUES (:symbol, :base, :quote)"
        ), {"symbol": symbol, "base": base_asset, "quote": quote_asset})
        result = conn.execute(text("SELECT symbol_id FROM symbol WHERE symbol = :symbol"), {"symbol": symbol}).fetchone()
        return result[0]

def get_or_create_interval(interval_name):
    seconds = INTERVAL_SECONDS[interval_name]
    with engine.begin() as conn:
        result = conn.execute(text("SELECT interval_id FROM intervals WHERE interval_name = :interval"), {"interval": interval_name}).fetchone()
        if result:
            return result[0]
        conn.execute(text(
            "INSERT INTO intervals (interval_name, seconds) VALUES (:interval, :seconds)"
        ), {"interval": interval_name, "seconds": seconds})
        result = conn.execute(text("SELECT interval_id FROM intervals WHERE interval_name = :interval"), {"interval": interval_name}).fetchone()
        return result[0]

def fetch_klines(symbol, interval, limit):
    url = f"{BASE_URL}{KLINE_ENDPOINT}"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()

def klines_to_dataframe(klines):
    columns = ["open_time","open","high","low","close","volume","close_time","quote_asset_volume",
               "number_of_trades","taker_buy_base_asset_volume","taker_buy_quote_asset_volume","ignore"]
    df = pd.DataFrame(klines, columns=columns)
    df["open_time"] = pd.to_datetime(df["open_time"], unit='ms')
    df["close_time"] = pd.to_datetime(df["close_time"], unit='ms')
    numeric_cols = ["open","high","low","close","volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    return df[["open_time","open","high","low","close","volume","close_time"]]

def save_klines(df, symbol_id, interval_id):
    df = df.copy()
    df["symbol_id"] = symbol_id
    df["interval_id"] = interval_id
    df["created_at"] = pd.Timestamp.now()
    df.to_sql("klines", engine, if_exists="append", index=False,
              dtype={
                  "open_time": DateTime(),
                  "close_time": DateTime(),
                  "open": Float(),
                  "high": Float(),
                  "low": Float(),
                  "close": Float(),
                  "volume": Float(),
                  "symbol_id": Integer(),
                  "interval_id": Integer(),
                  "created_at": DateTime()
              })

# --- ETL pipeline ---
def etl_pipeline():
    wait_for_postgres()
    create_tables()
    if not is_klines_empty():
        print("Table klines déjà alimentée, sortie du script.")
        return
    for symbol in SYMBOLS:
        symbol_id = get_or_create_symbol(symbol)
        for interval in INTERVALS:
            interval_id = get_or_create_interval(interval)
            print(f"Fetching {symbol} - {interval}")
            klines = fetch_klines(symbol, interval, LIMIT)
            df = klines_to_dataframe(klines)
            save_klines(df, symbol_id, interval_id)
    print("ETL terminé !")

if __name__ == "__main__":
    etl_pipeline()
