# Créer une fonction de récupération de données générique afin de pouvoir avoir les données de n’importe quel marché.
    # fonction de récupération des données de n'importe quel marché

# script pour streaming depuis websocket binance 
import websocket #sur bash importer : pip install websocket-client
import json
import os
import time 
import pandas as pd
from datetime import datetime
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StringType, DoubleType, TimestampType, BooleanType
import threading 

# Liste des symbols/marchés
SYMBOLS = ["btcusdt", "ethusdt", "bnbusdt", "solusdt"] # possibilité d'en ajouter
# Initialisation Spark
spark = SparkSession.builder \
    .appName("Stream_binance") \
    .master("local[*]") \
    .getOrCreate()  

schema = StructType() \
    .add("symbol", StringType()) \
    .add("open", DoubleType()) \
    .add("high", DoubleType()) \
    .add("low", DoubleType()) \
    .add("close", DoubleType()) \
    .add("volume", DoubleType()) \
    .add("timestamp", TimestampType()) \
    .add("closed", BooleanType())


# Création liste vide des données partagées entre threads
shared_data= []
lock = threading.Lock()  # Verrou pour la synchronisation des threads

# Fichier de sortie json
output_file = 'streaming_data.json'
if os.path.exists(output_file):
    os.remove(output_file)

# Fonction Websocket de callback pour la réception des messages
def on_message(ws, message):
    global shared_data
    data = json.loads(message)
    # Extraction des informations pertinentes
    try:# Création d'un dictionnaire pour la nouvelle donnée
        kline = data['k']
        new_data = {
            "symbol": kline['s'].lower(),
            "open": float(kline['o']),
            "high": float(kline['h']),
            "low": float(kline['l']),
            "close": float(kline['c']),
            "volume": float(kline['v']),
            "timestamp": datetime.fromtimestamp(kline['T'] / 1000.0), # Conversion en datetime
            "closed": bool(kline['x']) # Indique si la bougie est fermée ou non
        }
        with lock:
            shared_data.append(new_data)  # Ajout de la nouvelle donnée à la liste partagée

    except KeyError as e:
        print(f"KeyError: {e} in message {message}")
        


# Fonction de callback pour la gestion des erreurs
def on_error(ws, error):
    print(f"Error: {error}")    

# Fonction de callback pour la fermeture de la connexion
def on_close(ws, close_status_code, close_msg):
    print("### closed ###") 

# Fonction de callback pour l'ouverture de la connexion
def on_open(ws):
    print("Connection opened")  

# Fonction pour démarrer le WebSocket dans un thread séparé @trade
def start_websocket(symbol):
    ws_url = f"wss://stream.binance.com:9443/ws/{symbol}@kline_1m"  # Flux de données pour les trades en temps réel
    print(f"Starting WebSocket for {symbol} at {ws_url}")
    ws = websocket.WebSocketApp(
        ws_url,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open = on_open)
    ws.run_forever()


## Process_data avec Spark: Fonction pour traiter et afficher les données toutes les 10 secondes
def process_data():
    while True:
        time.sleep(10)  # Attendre 10 secondes
        with lock:
            if shared_data:
                # Création d'un DataFrame Spark à partir des données partagées
                rows = [Row(**data) for data in shared_data]
                df = spark.createDataFrame(rows, schema)
                df = df.filter(df.closed == True) # Ne garder que les bougies fermées/finalisées
                df.show(truncate=False)  # Affichage des données
                if not df.rdd.isEmpty(): # Vérifier si le DataFrame n'est pas vide avant d'écrire
                    df_filtered = df.select("symbol", "open", "high", "low", "close", "volume", "timestamp") # filter colonnes à sauvegarder
                    pandas_df = df_filtered.toPandas() #convertir en dataframe en pandas
                    pandas_df['timestamp'] = pandas_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S') # formatage datetime pour json
                    with open(output_file, 'a') as f:
                        pandas_df.to_json(output_file, orient='records', lines=True) # Enregistrement dans un fichier json
                shared_data.clear()  # Vider la liste après traitement


## Lancement des threads pour chaque symbole
# Démarrage des WebSockets pour chaque symbole dans des threads séparés
for symbol in SYMBOLS:
    threading.Thread(target=start_websocket, args=(symbol,)).start()


# Démarrage du traitement des données dans un thread séparé
threading.Thread(target=process_data).start()


## Boucle principale pour garder le script en cours d'exécution
try :
    while True:
        time.sleep(1)  
except KeyboardInterrupt:
    print("Arrêt du streaming...")
    spark.stop() # Arrêt de Spark proprement à la fin (ceci ne sera jamais atteint dans ce script)

# Note : Pour arrêter le script, utilisez Ctrl+C dans le terminal.
# Note : Ce script enregistre les données de streaming dans un fichier json nommé ''streaming_data.json'.
# Note : Assurez-vous d'avoir les bibliothèques nécessaires installées : websocket-client, pandas, pyspark.

# Note : Le script utilise des threads pour gérer le WebSocket et le traitement des données simultanément.
# Note : La synchronisation des threads est assurée à l'aide d'un verrou pour éviter les conflits d'accès aux données partagées.

# Note : Le script utilise la bibliothèque websocket-client pour gérer les connexions WebSocket.
# Note : La bibliothèque pandas est utilisée pour la manipulation des données avant l'enregistrement dans un fichier CSV.
# Note : La bibliothèque pyspark est utilisée pour le traitement des données en utilisant Spark.


