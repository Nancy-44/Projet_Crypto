import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

# ============================================================

# Étape 1 : Chargement & Préparation

# ============================================================

df = pd.read_csv("dataset_export_1h_4h_1d.csv")
interval_mapping = {2: 60, 3: 4*60, 4: 24*60}
symbol_mapping = {1: "BTCUSDT", 2: "ETHUSDT", 3: "BNBUSDT", 4: "SOLUSDT"}
df["interval_min"] = df["interval_id"].map(interval_mapping)
df["symbol_label"] = df["symbol_id"].map(symbol_mapping)

horizon = 10
threshold = 0.01
df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1
df["target"] = np.where(df["future_return"] > threshold, 1,
np.where(df["future_return"] < -threshold, 0, np.nan))
df.dropna(subset=["target"], inplace=True)
df.dropna(inplace=True)

# ============================================================

# Étape 2 : Calcul unique des indicateurs

# ============================================================

def compute_RSI(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_indicators(df):
    df = df.copy()
    df["RSI"] = compute_RSI(df["close"])
    df["MA_10"] = df["close"].rolling(10, min_periods=10).mean()
    df["MA_50"] = df["close"].rolling(50, min_periods=50).mean()
    df["STD_20"] = df["close"].rolling(20, min_periods=20).std()
    df["RET"] = df["close"].pct_change()
    df["VOLAT"] = df["RET"].rolling(10, min_periods=10).std()
    for lag in [1, 2, 3]:
        df[f"close_lag{lag}"] = df["close"].shift(lag)
        df[f"volume_lag{lag}"] = df["volume"].shift(lag)
        df.dropna(inplace=True)
    return df

df = compute_indicators(df)

# ============================================================

# Étape 3 : Boucle sur symboles avec optimisation

# ============================================================

results = {}
symbols = df["symbol_label"].unique()

for symbol in symbols:
    data = df[df["symbol_label"] == symbol].copy()
    if len(data) < 300:
        continue


    print(f"\n=== {symbol} ===")
    features = [c for c in data.columns if c not in ["target","symbol_label","symbol_id","interval_id","time","future_return"]]
    X = data[features].select_dtypes(include=[np.number])
    y = data["target"]

    split_idx = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Gestion du déséquilibre
    classes = np.unique(y_train)
    class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_dict = dict(zip(classes, class_weights))
    scale_pos_weight = weight_dict[0] / weight_dict[1]

    tuned_models = {
        "RandomForest_opt": (RandomForestClassifier(n_estimators=200, max_depth=8, max_features="sqrt", random_state=42, class_weight="balanced"), {
            "n_estimators": [100, 300, 500],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "max_features": ["sqrt", "log2"]
        }),
        "XGBoost_opt": (XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,eval_metric="logloss", use_label_encoder=False, random_state=42, scale_pos_weight=scale_pos_weight, verbosity=0), {
            "n_estimators": [200, 400],
            "max_depth": [3, 5, 8],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        }),
        "LightGBM_opt": (LGBMClassifier(random_state=42, scale_pos_weight=scale_pos_weight, verbose=-1), {
            "num_leaves": [15, 31, 63],
            "max_depth": [3, 6, 9, -1],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [200, 400],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        })
    }

    scores = {}
    for name, (model, param_grid) in tuned_models.items():
        search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, cv=3,
                                scoring="f1", n_jobs=-1, random_state=42)
        search.fit(X_train_scaled, y_train)
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test_scaled)
        scores[name] = {"accuracy": accuracy_score(y_test, y_pred), "f1": f1_score(y_test, y_pred)}
        print(f"{name:18s} | Acc(opt): {scores[name]['accuracy']:.3f} | F1(opt): {scores[name]['f1']:.3f}")

    best_model_name = max(scores, key=lambda x: scores[x]["f1"])
    results[symbol] = {**scores[best_model_name], "best_model": best_model_name}


print("\n=== Résumé global ===")
for sym, res in results.items():
    print(f"{sym:10s} → {res['best_model']:15s} | Acc {res['accuracy']:.2f} | F1 {res['f1']:.2f}")
