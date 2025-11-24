from pymongo import MongoClient
import psycopg2
import os

def get_postgres_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        database=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD")
    )

def get_mongo_collection():
    mongo_host = os.getenv("MONGO_HOST")
    mongo_port = int(os.getenv("MONGO_PORT"))
    mongo_user = os.getenv("MONGO_USER")
    mongo_pass = os.getenv("MONGO_PASSWORD")
    mongo_db = os.getenv("MONGO_DB")

    # Construire la URI selon auth
    if mongo_user and mongo_pass:
        uri = f"mongodb://{mongo_user}:{mongo_pass}@{mongo_host}:{mongo_port}/?authSource=admin"
    else:
        uri = f"mongodb://{mongo_host}:{mongo_port}/"

    client = MongoClient(uri)
    db = client[mongo_db]
     # Tester la connexion
    try:
        client.admin.command("ping")
        print(f" Connexion MongoDB réussie sur {mongo_host}:{mongo_port}")
    except ServerSelectionTimeoutError as err:
        print(f" Impossible de se connecter à MongoDB: {err}")
        raise err
    return db["klines"]
