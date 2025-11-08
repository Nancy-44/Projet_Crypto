import pandas as pd
import psycopg2

# Connexion à PostgreSQL
conn = psycopg2.connect(
    host="54.78.190.72",
    port=5432,
    dbname="crypto_db",
    user="ubuntu",
    password="postgres"
)

# Requête SQL
query = "SELECT * FROM klines WHERE interval_id IN (2);"

# Lire les données dans un DataFrame
df = pd.read_sql(query, conn)

# Export en CSV
df.to_csv("dataset_export_1h.csv", index=False)

conn.close()