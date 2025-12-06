import streamlit as st
import pandas as pd
from api_client import get_prediction, get_latest, get_historical
import plotly.graph_objects as go


st.set_page_config(page_title="Crypto Prediction Dashboard", layout="wide")
st.title(" Crypto Prediction Dashboard")
st.subheader("Prédiction + Dernière valeur temps réel + Historique (tableau)")
DEFAULT_INTERVAL = "1 min"   # valeur affichée

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
    # VALIDATION DES DONNÉES
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

    
    # ------------------------------------------------------------
    # COLONNES DASHBOARD
    # ------------------------------------------------------------
    col_left, col_right = st.columns(2)

    # --------------------------
    # AFFICHAGE PREDICTION et PROBABILITÉS
    # --------------------------
    with col_left:
        st.markdown(f"""
        ### Prédiction du modèle — {symbol}
        **Signal : `{prediction['prediction']}`**  
        - Probabilité BUY : **{prediction['prob_buy']:.3f}**  
        - Probabilité SELL : **{prediction['prob_sell']:.3f}**  
        - Timestamp : `{prediction['timestamp']}`
        """)

    
    # ------------------------------------------------------------
    # COLONNE GAUCHE : DERNIÈRE VALEUR
    # ------------------------------------------------------------
    with col_left:
        st.markdown("### Dernière valeur streaming (/latest)")
        st.markdown(f"""
        - Close : **{latest['close']}**  
        - Open : {latest.get('open', 'N/A')}  
        - High : {latest.get('high', 'N/A')}  
        - Low : {latest.get('low', 'N/A')}  
        - Volume : {latest.get('volume', 'N/A')}  
        - Close time : `{latest['close_time']}`
        """)

        st.markdown(f"**Intervalle utilisé : {DEFAULT_INTERVAL}**") # Affichage uniquement pas de liste déroulante


    # ------------------------------------------------------------
    # COLONNE DROITE : CANDLE CHART
    # ------------------------------------------------------------
    with col_right:
        st.markdown("### Candle Chart (données historiques)")

        if isinstance(historical, list) and historical:
            df = pd.DataFrame(historical)
            df["close_time"] = pd.to_datetime(df["close_time"])

            # Ligne médiane (mid-price)
            df["median"] = (df["high"] + df["low"]) / 2

            fig = go.Figure()

            # --- CANDLESTICK ---
            fig.add_trace(go.Candlestick(
                x=df["close_time"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="Bougies"
            ))

            # --- LIGNE MÉDIANE ---
            fig.add_trace(go.Scatter(
                x=df["close_time"],
                y=df["median"],
                mode="lines",
                line=dict(width=1.2, dash="dot"),
                name="Ligne médiane"
            ))

            fig.update_layout(
                height=600,
                title_text=f"Candlestick {symbol} (historique)",
                xaxis_title="Date",
                yaxis_title="Prix",
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Aucune donnée historique disponible pour afficher le candle chart.")


        
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
