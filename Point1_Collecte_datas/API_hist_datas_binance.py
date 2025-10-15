import requests
import pandas as pd

#Paramètres de l'API Binance
BASE_URL = "https://api.binance.com"
KLINE_ENDPOINT = "/api/v3/klines"
SYMBOL = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"] # Liste des symboles disponibles à compléter
INTERVAL = "1h"
LIMIT = 20 # max 1000 transactions


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

# Fonction principale pour obtenir les données historiques
def get_historical_data(symbol=SYMBOL, interval=INTERVAL, limit=LIMIT):
    klines = fetch_klines(symbol, interval, limit)
    df = klines_to_dataframe(klines)
    return df

# Exécution directe pour tester la fonction
if __name__ == "__main__":
    for symbol in SYMBOL:
        print(f"Récupération des données pour le symbole: {symbol}")
        df = get_historical_data(symbol) # Utiliser le symbole par défaut
        print(df.head())# Afficher les premières lignes du DataFrame 
        print(df.dtypes) # Afficher les types de données des colonnes
        print(df.describe()) # Afficher les statistiques descriptives du DataFrame
        print(df.info()) # Afficher les informations sur le DataFrame
        print(df.columns) # Afficher les noms des colonnes du DataFrame
        print(df.shape) # Afficher la forme du DataFrame
        print(df.tail()) # Afficher les dernières lignes du DataFrame
        print(df.isnull().sum()) # Vérifier les valeurs nulles dans le DataFrame
        df.to_csv(f"{symbol}_klines.csv", index=False) # Export vers un fichier CSV
        print(f"Données exportées vers {symbol}_klines.csv")
        
