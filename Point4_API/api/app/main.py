from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from .db import get_postgres_conn, get_mongo_collection
from .auth import authenticate
from scipy.stats import ks_2samp

app = FastAPI(title="Crypto Prediction API")

# Chargement des modèles ML
MODELS = {
    "BTCUSDT": "/ml/models/BTCUSDT.pkl",
    "ETHUSDT": "/ml/models/ETHUSDT.pkl",
    "BNBUSDT": "/ml/models/BNBUSDT.pkl",
    "SOLUSDT": "/ml/models/SOLUSDT.pkl",
}

# Models Pydantec
class PredictRequest(BaseModel):
    symbol: str
    interval: str = "15m" # 15m (par défaut), possibilité de changer avec 1h, 4h, 1d

class DriftRequest(BaseModel):
    symbol: str
    interval: str = "15m"
    window: int = 60
    metric: str = "ks"   # pour le moment seul KS-test est implémenté

# --- endpoint HEALTH ---
@app.get("/health")
def health():
    """Vérifie que l'API est fonctionnelle."""
    return {"message": "L'API est fonctionnelle."}

# --- endpoint HISTORICAL ---
@app.get("/historical")
def get_historical(symbol: str, limit: int = 10, auth: bool = Depends(authenticate)):
    """Récupère les dernières bougies stockées dans PostgreSQL."""
    conn = get_postgres_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.symbol, k.open_time, k.close_time, k.open, k.high, k.low, k.close, k.volume
        FROM klines k
        JOIN symbol s ON k.symbol_id = s.symbol_id
        WHERE s.symbol = %s
        ORDER BY k.open_time DESC
        LIMIT %s
    """, (symbol, limit))

    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail="No historical data found.")

    result = [
        {
            "symbol": row[0],
            "open_time": row[1],
            "close_time": row[2],
            "open": row[3],
            "high": row[4],
            "low": row[5],
            "close": row[6],
            "volume": row[7]
        }
        for row in rows
    ]
    return result

# --- endpoint LATEST ---
@app.get("/latest")
def get_latest(symbol: str, auth: bool = Depends(authenticate)):
    """Récupère la dernière bougie (MongoDB)."""
    collection = get_mongo_collection()
    # pour récupérer le dernier document du symbole trié par index du plus récent.
    doc = collection.find_one({"symbol": symbol}, sort=[("_id", -1)]) 
    if not doc:
        raise HTTPException(status_code=404, detail="No live data found")
    doc["_id"] = str(doc["_id"])
    return doc


# Fonctions pour features : RSI, moyennes mobiles, volatilité, décalages.

def compute_RSI(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_features(df):
    df = df.copy()
    df["RSI"] = compute_RSI(df["close"])
    df["MA_10"] = df["close"].rolling(10).mean()
    df["MA_50"] = df["close"].rolling(50).mean()
    df["STD_20"] = df["close"].rolling(20).std()
    df["RET"] = df["close"].pct_change()
    df["VOLAT"] = df["RET"].rolling(10).std()
    for lag in [1,2,3]:
        df[f"close_lag{lag}"] = df["close"].shift(lag)
        df[f"volume_lag{lag}"] = df["volume"].shift(lag)
    df.dropna(inplace=True)
    return df

# --- PREDICT ---
@app.post("/predict")
def predict(req: PredictRequest, auth: bool = Depends(authenticate)):
    """Calcule une prédiction Achat(Buy) ou vente(sell) avec une probabilité."""
    # Vérifie que le modèle existe
    if req.symbol not in MODELS:
        return {"error": "model not available"}

    model_path = MODELS[req.symbol]
    model = joblib.load(model_path)

    # Récupérer les 60 dernières bougies fermées depuis MongoDB
    collection = get_mongo_collection()
    df = pd.DataFrame(list(
        collection.find({"symbol": req.symbol, "closed": True}).sort("close_time", -1).limit(60)
    ))
    if df.empty:
        raise HTTPException(status_code=404, detail="No data available for prediction")

    df = df.sort_values("close_time")

    # Calcul des 17 features pour le modèle
    df_feat = compute_features(df)

    feature_cols = [
        "open","high","low","close","volume",
        "RSI","MA_10","MA_50","STD_20","RET","VOLAT",
        "close_lag1","close_lag2","close_lag3",
        "volume_lag1","volume_lag2","volume_lag3"
    ]
    if not all(col in df_feat.columns for col in feature_cols):
        raise HTTPException(status_code=400, detail="Not enough data to compute features")

    X = df_feat[feature_cols].iloc[-1:].values  # dernière ligne = dernière bougie préparée

    # Normalisation si un fichier scaler existe en .pkl (sauvegardé lors de l'entraînement)
    scaler_path = f"/models/{req.symbol}_scaler.pkl"
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X = scaler.transform(X)

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    return {
        "symbol": req.symbol,
        "prediction": "BUY" if pred == 1 else "SELL",
        "prob_buy": float(proba[1]),
        "prob_sell": float(proba[0]),
        "timestamp": df_feat["close_time"].iloc[-1].isoformat()
    }

# ---- Endpoint DRIFT - KS-test ------
@app.post("/drift")
def check_drift(req: DriftRequest, auth: bool = Depends(authenticate)):

    if req.symbol not in MODELS:
        raise HTTPException(status_code=400, detail="model not available")

    # ---------- 1. Données historiques (PostgreSQL)
    conn = get_postgres_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT k.open, k.high, k.low, k.close, k.volume
        FROM klines k
        JOIN symbol s ON k.symbol_id = s.symbol_id
        WHERE s.symbol = %s
        ORDER BY k.open_time DESC
        LIMIT 1000
    """, (req.symbol,))
    hist_rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if not hist_rows:
        raise HTTPException(status_code=404, detail="No historical data available")

    df_hist = pd.DataFrame(hist_rows, columns=["open","high","low","close","volume"])

    # ---------- 2. Données récentes (MongoDB)
    collection = get_mongo_collection()
    df_recent = pd.DataFrame(list(
        collection.find({"symbol": req.symbol, "closed": True}).sort("close_time", -1).limit(req.window)
    ))

    if df_recent.empty:
        raise HTTPException(status_code=404, detail="No recent data available")

    df_recent = df_recent[["open","high","low","close","volume"]]

    # ---------- 3. Features
    df_hist_feat = compute_features(df_hist)
    df_recent_feat = compute_features(df_recent)

    if df_hist_feat.empty or df_recent_feat.empty:
        raise HTTPException(status_code=400, detail="Not enough data to compute features for drift")

    # ---------- 4. Calcul KS-test
    drift_results = {}
    for col in df_hist_feat.columns:
        stat, p_value = ks_2samp(df_hist_feat[col], df_recent_feat[col])
        drift_results[col] = {
            "ks_stat": float(stat),
            "p_value": float(p_value)
        }

    # Détection si un drift est présent (p < 0.05)
    drift_detected = any(v["p_value"] < 0.05 for v in drift_results.values())

    return {
        "symbol": req.symbol,
        "drift_detected": drift_detected,
        "metric": "ks",
        "window": req.window,
        "results": drift_results
    }
