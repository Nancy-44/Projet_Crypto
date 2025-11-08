import requests
import pandas as pd

from sqlalchemy import create_engine, text
from sqlalchemy.types import Float, DateTime, Integer, String

engine = create_engine('postgresql://ubuntu:postgres@localhost:5432/crypto_db') # Connexion à la base de données PostgreSQL

#Paramètres de l'API Binance
BASE_URL = "https://api.binance.com"
KLINE_ENDPOINT = "/api/v3/klines"
SYMBOL = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"] # Liste des symboles disponibles à compléter
INTERVALS = ["1m", "5m", "15m", "1h", "4h", "1d"]
LIMIT = 20 # max 1000 transactions
INTERVAL_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400
}
# Création des tables dans PostgreSQL si elles n'existent pas
def create_table_if_not_exists():
    with engine.connect() as connection:
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS KLINES (
                klines_id SERIAL PRIMARY KEY,
                symbol_id INTEGER,
                interval_id INTEGER,
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
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS SYMBOL (
                symbol_id SERIAL PRIMARY KEY,
                symbol VARCHAR(10) UNIQUE,
                base_asset VARCHAR(10),
                quote_asset VARCHAR(10)
            );
        """))
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS INTERVAL (
                interval_id SERIAL PRIMARY KEY,
                interval_name VARCHAR(10) UNIQUE,
                symbol_id INT REFERENCES SYMBOL(symbol_id),
                seconds INTEGER
            );
        """))

# Fonctions d'insertion dans les tables SYMBOL et INTERVAL
def insert_symbol_if_not_exists(symbol_name, base_asset, quote_asset): 
    with engine.connect() as connection:
        result = connection.execute(
            "SELECT symbol_id FROM SYMBOL WHERE symbol = %s", (symbol_name,)
        ).fetchone()
        if result is None:
            connection.execute(
                "INSERT INTO SYMBOL (symbol, base_asset, quote_asset) VALUES (%s, %s, %s)",
                (symbol_name, base_asset, quote_asset)
            )

def insert_interval_if_not_exists(interval_name, symbol_id, seconds):
    with engine.connect() as connection:
        result = connection.execute(
            "SELECT interval_id FROM INTERVAL WHERE interval_name = %s AND symbol_id = %s",
            (interval_name, symbol_id)
        ).fetchone()
        if result is None:
            connection.execute(
                "INSERT INTO INTERVAL (interval_name, symbol_id, seconds) VALUES (%s, %s, %s)",
                (interval_name, symbol_id, seconds)
            )                       




# Fonction pour récupérer les données de chandeliers (klines) depuis l'API Binance
def fetch_klines(symbol, interval, limit):
    url = f"{BASE_URL}{KLINE_ENDPOINT}"
    params = {
        "symbol": symbol,  
        "interval": interval,
        "limit": limit
    }
    response = requests.get(url, params=params)
    response.raise_for_status()  # Vérifie que la requête a réussi
    return response.json()


# Fonction pour convertir les données de klines en DataFrame pandas
def klines_to_dataframe(klines):
    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ]
    df = pd.DataFrame(klines, columns=columns)
    df["open_time"] = pd.to_datetime(df["open_time"], unit='ms') # Convertir en datetime
    df["close_time"] = pd.to_datetime(df["close_time"], unit='ms') # Convertir en datetime
    
    numeric_columns = ["open", "high", "low", "close", "volume"]
    df[numeric_columns] = df[numeric_columns].astype(float) # Convertir en float
    df = df[["open_time", "open", "high", "low", "close", "volume", "close_time"]] # Garder uniquement les colonnes pertinentes
    return df

# Fonction pour sauvegarder les données dans PostgreSQL
def save_to_postgres(df, symbol, interval):
    table_name= "KLINES"
    df["symbol"]= symbol
    df["interval"]= interval
    df["created_at"]= pd.Timestamp.now()

    df.to_sql(
        "klines", 
        engine, 
        if_exists='append', 
        index=False,
        dtype={
            "open_time": DateTime(),
            "close_time": DateTime(),
            "open": Float(),
            "high": Float(),
            "low": Float(),
            "close": Float(),
            "volume": Float(),
            "symbol_id": Integer(),
            "interval_Id": Integer(),
            "created_at": DateTime() }
    )

def etl_pipeline():
    create_table_if_not_exists()
    for symbol in SYMBOL:
        base_asset = symbol[:-4]  # Supposons que le symbole se termine par 'USDT'
        quote_asset = symbol[-4:]
        insert_symbol_if_not_exists(symbol, base_asset, quote_asset)
        
        for interval in INTERVALS:
            seconds = INTERVAL_SECONDS[interval]
            interval_id = insert_interval_if_not_exists(interval, symbol, seconds)
            
            print(f"Récupération des données pour le symbole: {symbol} - {interval}")
            klines = fetch_klines(symbol, interval, LIMIT)
            df = klines_to_dataframe(klines)
            save_to_postgres(df, symbol, interval_id)

if __name__ == "__main__":
    etl_pipeline()

           
        
