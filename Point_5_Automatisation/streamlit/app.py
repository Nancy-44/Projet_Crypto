import streamlit as st
import pandas as pd
from api_client import get_prediction, get_latest, get_historical

st.set_page_config(page_title="Crypto Prediction Dashboard", layout="wide")
st.title(" Crypto Prediction Dashboard")
st.subheader("Prédiction + Dernière valeur temps réel + Historique (tableau)")

# --- Choix du symbole ---
symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
symbol = st.selectbox("Choisis un symbole :", symbols)

if st.button("Charger les données"):

    # --------------------------
    # APPELS API
    # --------------------------
    prediction = get_prediction(symbol)
    latest = get_latest(symbol)
    historical = get_historical(symbol)

    # --------------------------
    # VALIDATION
    # --------------------------
    required_keys = ["symbol", "prediction", "prob_buy", "prob_sell", "timestamp"]
    if any(k not in prediction for k in required_keys):
        st.error(" Réponse invalide du endpoint /predict")
        st.json(prediction)
        st.stop()

    if "error" in latest:
        st.error(" Erreur /latest")
        st.json(latest)
        st.stop()

    if "error" in historical:
        st.error(" Erreur /historical")
        st.json(historical)
        st.stop()

    # --------------------------
    # AFFICHAGE PROBABILITÉS
    # --------------------------
    st.markdown(f"""
    ### Prédiction du modèle — {symbol}
    **Signal : `{prediction['prediction']}`**  
    - Probabilité BUY : **{prediction['prob_buy']:.3f}**  
    - Probabilité SELL : **{prediction['prob_sell']:.3f}**  
    - Timestamp : `{prediction['timestamp']}`
    """)

    # --------------------------
    # AFFICHAGE DERNIÈRE VALEUR
    # --------------------------
    st.markdown(f"""
    ### Dernière valeur streaming (/latest)
    - Close : **{latest['close']}**  
    - Open : {latest.get('open', 'N/A')}  
    - High : {latest.get('high', 'N/A')}  
    - Low : {latest.get('low', 'N/A')}  
    - Volume : {latest.get('volume', 'N/A')}  
    - Close time : {latest['close_time']}
    """)

    # --------------------------
    # AFFICHAGE DONNÉES HISTORIQUES
    # --------------------------
    if isinstance(historical, list) and historical:
        df = pd.DataFrame(historical)
        df["close_time"] = pd.to_datetime(df["close_time"])
        st.markdown("### Données historiques")
        st.dataframe(df)
    else:
        st.info("Aucune donnée historique disponible.")

    # --------------------------
    # DONNÉES BRUTES (expander)
    # --------------------------
    with st.expander("Voir toutes les données brutes API"):
        st.json({
            "prediction": prediction,
            "latest": latest,
            "historical": historical
        })
