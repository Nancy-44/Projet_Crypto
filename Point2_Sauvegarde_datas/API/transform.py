## Transforme les fichiers JSON bruts en CSV nettoyés.

import pandas as pd
import json
import os

def transform_klines_file(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    
    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ]
    # Converti les timestamp en datetime
    df = pd.DataFrame(data, columns=columns)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    # Converti les colonnes numériques en float
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    #Sélection des colonnes utiles pour csv
    df = df[["open_time", "open", "high", "low", "close", "volume", "close_time"]]
    return df

def run_transformation():
    os.makedirs("data/processed", exist_ok=True)
    
    raw_dir = "data/raw"
    processed_dir = "data/processed"

    if not os.path.exists(raw_dir):
        print(f"Le dossier {raw_dir} n'existe pas !")
        return

    for file in os.listdir(raw_dir):
        if file.endswith(".json"):
            filepath = os.path.join(raw_dir, file)
            df = transform_klines_file(filepath)

            csv_path = os.path.join(processed_dir, file.replace(".json", ".csv"))
            df.to_csv(csv_path, index=False)
            print(f"Transformé : {csv_path}")


if __name__ == "__main__":
    run_transformation()