# Script de prétraitement des données sortant du streaming et de l'API historique

import pandas as pd
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, DoubleType, TimestampType
from pyspark.sql import Row


# Fonction pour prétraiter les données de l'API historique
def preprocess_historical_data(df):
    """Prétraite les données historiques pour les rendre compatibles avec les données de streaming."""
    # Convertir les timestamps en datetime
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    
    # Renommer les colonnes pour correspondre aux données de streaming
    df.rename(columns={
        'open_time': 'timestamp',
        'close': 'price',
        'volume': 'quantity'
    }, inplace=True)
    
    # Ajouter une colonne 'symbol' avec une valeur par défaut (à ajuster selon le symbole utilisé)
    df['symbol'] = "BTCUSDT"  # Exemple, à modifier selon le symbole utilisé
    
    # Sélectionner uniquement les colonnes nécessaires
    df = df[['symbol', 'price', 'quantity', 'timestamp']]
    
    return df


# Fonction pour prétraiter les données de streaming
def preprocess_streaming_data(shared_data):
    """Prétraite les données de streaming pour les rendre compatibles avec les données historiques."""
    # Convertir la liste de dictionnaires en DataFrame pandas
    df = pd.DataFrame(shared_data)
    
    # Convertir les timestamps en datetime si nécessaire
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df

