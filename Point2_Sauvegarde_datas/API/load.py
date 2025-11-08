
from sqlalchemy import (text,Table, MetaData, Column, Integer, Float, DateTime)
import pandas as pd
import os
from config import engine, SYMBOLS, INTERVAL_SECONDS

def create_tables():
    """Création des tables PostgreSQL avec clés primaires et étrangères"""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS SYMBOL (
                symbol_id SERIAL PRIMARY KEY,
                symbol_name VARCHAR(10) UNIQUE NOT NULL,
                base_asset VARCHAR(10),
                quote_asset VARCHAR(10)
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS INTERVAL (
                interval_id SERIAL PRIMARY KEY,
                interval_name VARCHAR(10) UNIQUE NOT NULL,
                seconds INTEGER
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS KLINES (
                klines_id SERIAL PRIMARY KEY,
                symbol_id INT REFERENCES SYMBOL(symbol_id),
                interval_id INT REFERENCES INTERVAL(interval_id),
                open_time TIMESTAMP NOT NULL,
                close_time TIMESTAMP,
                open FLOAT,
                high FLOAT,
                low FLOAT,
                close FLOAT,
                volume FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
    print("Tables créées ou déjà existantes.")


def populate_reference_tables():
    """Insère les symboles et intervalles dans leurs tables respectives"""
    with engine.begin() as conn:
        # SYMBOL
        for s in SYMBOLS:
            base, quote = s[:-4], s[-4:]
            conn.execute(
                text("""
                    INSERT INTO SYMBOL (symbol_name, base_asset, quote_asset)
                    VALUES (:s, :b, :q)
                    ON CONFLICT (symbol_name) DO NOTHING;
                """),
                {"s": s, "b": base, "q": quote}
            )

        # INTERVAL
        for name, sec in INTERVAL_SECONDS.items():
            conn.execute(
                text("""
                    INSERT INTO INTERVAL (interval_name, seconds)
                    VALUES (:n, :s)
                    ON CONFLICT (interval_name) DO NOTHING;
                """),
                {"n": name, "s": sec}
            )
    print("Tables de référence SYMBOL et INTERVAL peuplées.")


def load_klines_data():
    """Charge les fichiers transformés CSV dans la table KLINES via SQLAlchemy insert()"""
    metadata = MetaData()
    klines_table = Table(
        "klines", metadata,
        Column("klines_id", Integer, primary_key=True),
        Column("symbol_id", Integer),
        Column("interval_id", Integer),
        Column("open_time", DateTime),
        Column("close_time", DateTime),
        Column("open", Float),
        Column("high", Float),
        Column("low", Float),
        Column("close", Float),
        Column("volume", Float),
        Column("created_at", DateTime)
    )

    for file in os.listdir("data/processed"):
        if not file.endswith(".csv"):
            continue

        symbol_name, interval_name = file.replace(".csv", "").split("_")
        df = pd.read_csv(f"data/processed/{file}")

        # Conversion des colonnes temporelles
        df['open_time'] = pd.to_datetime(df['open_time']).dt.tz_localize(None)
        df['close_time'] = pd.to_datetime(df['close_time']).dt.tz_localize(None)

        # Conversion numérique
        numeric_cols = ["open", "high", "low", "close", "volume"]
        df[numeric_cols] = df[numeric_cols].astype(float)

        with engine.begin() as conn:
            symbol_id = conn.execute(
                text("SELECT symbol_id FROM SYMBOL WHERE symbol_name = :s"),
                {"s": symbol_name}
            ).scalar()

            interval_id = conn.execute(
                text("SELECT interval_id FROM INTERVAL WHERE interval_name = :i"),
                {"i": interval_name}
            ).scalar()

            if symbol_id is None or interval_id is None:
                print(f"Symbol ou interval non trouvé pour {file}")
                continue

            df['symbol_id'] = int(symbol_id)
            df['interval_id'] = int(interval_id)

            # Colonnes finales à insérer
            records = df[[
                "symbol_id", "interval_id", "open_time", "close_time",
                "open", "high", "low", "close", "volume"
            ]].to_dict(orient='records')

            conn.execute(klines_table.insert(), records)

        print(f" Données chargées : {file}")


if __name__ == "__main__":
    create_tables()
    populate_reference_tables()
    load_klines_data()