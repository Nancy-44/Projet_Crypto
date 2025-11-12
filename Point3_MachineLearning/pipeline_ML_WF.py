import pandas as pd
import numpy as np
import warnings
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# ===========================
# Paramètres principaux
# ===========================
horizon = 10           # horizon de prédiction
threshold = 0.05      # seuil de variation pour target binaire

interval_mapping = {2: 60, 3: 240, 4: 1440}  # minutes
symbol_mapping = {1: "BTCUSDT", 2: "ETHUSDT", 3: "BNBUSDT", 4: "SOLUSDT"}

# ===========================
# Chargement dataset
# ===========================
df = pd.read_csv("dataset_export_1h_4h_1d.csv")
df["interval_min"] = df["interval_id"].map(interval_mapping)
df["symbol_label"] = df["symbol_id"].map(symbol_mapping)

# ===========================
# Target
# ===========================
df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1
df["target"] = np.where(df["future_return"] > threshold, 1,
                 np.where(df["future_return"] < -threshold, 0, np.nan))
df.dropna(subset=["target"], inplace=True)

# ===========================
# Fonctions features
# ===========================
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

# ===========================
# MLflow experiment
# ===========================
mlflow.set_experiment("crypto_signals_v1")

# ===========================
# Boucle sur les symboles
# ===========================
results = {}
symbols = df["symbol_label"].unique()

for symbol in symbols:
    data = df[df["symbol_label"] == symbol].copy()
    if len(data) < 300:
        continue

    print(f"\n=== {symbol} ===")

    # Séparer train/test avant feature engineering
    raw_features = data[["open","high","low","close","volume"]].copy()
    raw_target = data["target"]
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        raw_features, raw_target, test_size=0.3, shuffle=False
    )

    train_data = X_train_raw.copy()
    train_data["target"] = y_train
    train_data = compute_indicators(train_data)
    X_train = train_data.drop(columns=["target"])
    y_train = train_data["target"]

    test_data = X_test_raw.copy()
    test_data["target"] = y_test
    test_data = compute_indicators(test_data)
    X_test = test_data.drop(columns=["target"])
    y_test = test_data["target"]

    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Gestion déséquilibre
    classes = np.unique(y_train)
    class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_dict = dict(zip(classes, class_weights))
    scale_pos_weight = weight_dict[0] / weight_dict[1]

    # ===========================
    # Modèles
    # ===========================
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=150, max_depth=6, max_features="sqrt",
                                               random_state=42, class_weight="balanced"),
        "XGBoost": XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                                 use_label_encoder=False, random_state=42, scale_pos_weight=scale_pos_weight, verbosity=0),
        "LightGBM": LGBMClassifier(num_leaves=31, max_depth=5, learning_rate=0.05,
                                   n_estimators=200, subsample=0.8, colsample_bytree=0.8,
                                   scale_pos_weight=scale_pos_weight, random_state=42, verbosity=-1)
    }

    scores = {}

    # ===========================
    # Boucle MLflow pour base + OPT
    # ===========================
    for name, model in models.items():
        with mlflow.start_run(run_name=f"{symbol}_{name}_BASE") as run:
            mlflow.set_tag("symbol", symbol)
            mlflow.set_tag("model_type", name)
            mlflow.set_tag("horizon", horizon)
            mlflow.set_tag("threshold", threshold)

            # entraînement
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            # scores
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1", f1)

            # log modèle
            mlflow.sklearn.log_model(model, "model")

            print(f"[BASE] {name:12s} | Acc: {acc:.3f} | F1: {f1:.3f}")

    # ===========================
    # Hyperparam optimization
    # ===========================
    param_rf = {"n_estimators":[100,200,300], "max_depth":[3,5,6,8], "min_samples_split":[2,5], "max_features":["sqrt","log2"]}
    param_xgb = {"n_estimators":[100,200], "max_depth":[3,4,5], "learning_rate":[0.01,0.05], "subsample":[0.7,0.8],
                 "colsample_bytree":[0.7,0.8], "scale_pos_weight":[scale_pos_weight]}
    param_lgb = {"num_leaves":[15,31], "max_depth":[3,5], "learning_rate":[0.01,0.05],
                 "n_estimators":[100,200], "subsample":[0.7,0.8], "colsample_bytree":[0.7,0.8], "scale_pos_weight":[scale_pos_weight]}

    tuned_models = {
        "RandomForest_OPT": (RandomForestClassifier(random_state=42, class_weight="balanced"), param_rf),
        "XGBoost_OPT": (XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42), param_xgb),
        "LightGBM_OPT": (LGBMClassifier(random_state=42, verbosity=-1), param_lgb)
    }

    for name, (model, param_grid) in tuned_models.items():
        with mlflow.start_run(run_name=f"{symbol}_{name}") as run:
            mlflow.set_tag("symbol", symbol)
            mlflow.set_tag("model_type", name)
            mlflow.set_tag("horizon", horizon)
            mlflow.set_tag("threshold", threshold)

            search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, cv=3,
                                        scoring="f1", n_jobs=-1, random_state=42)
            search.fit(X_train_scaled, y_train)
            best_model = search.best_estimator_
            y_pred = best_model.predict(X_test_scaled)

            # scores
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1", f1)

            # log modèle
            mlflow.sklearn.log_model(best_model, "model")

            print(f"[OPT] {name:12s} | Acc: {acc:.3f} | F1: {f1:.3f}")
            # sauvegarde local si besoin
            joblib.dump(best_model, f"{symbol}_{name}.pkl")

            scores[name] = {"accuracy": acc, "f1": f1}

    # Meilleur modèle
    best_model_name = max(scores, key=lambda x: scores[x]["f1"])
    results[symbol] = {**scores[best_model_name], "best_model": best_model_name}

# ===========================
# Résumé global
# ===========================
print("\n=== Résumé global ===")
for sym, res in results.items():
    print(f"{sym:10s} → {res['best_model']:18s} | Acc {res['accuracy']:.3f} | F1 {res['f1']:.3f}")
