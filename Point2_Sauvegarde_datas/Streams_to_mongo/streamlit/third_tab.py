import streamlit as st
import pandas as pd
import numpy as np

import add_datas.postgres.config as config
from sqlalchemy import create_engine

DATABASE_URL = "postgresql://ubuntu:postgres@localhost:5432/crypto_db"
engine = create_engine(DATABASE_URL)

title = "Data from Postgresql"
sidebar_name = title


def run():

    st.title(title)

    st.markdown(
        """
        This is the data ingested in postgresql
        """
    )
    query = "SELECT * FROM klines;"
    df = pd.read_sql(query, engine)
    #st.write(pd.DataFrame(np.random.randn(100, 4), columns=list("ABCD")))
    #st.write(df.head())

    interval = st.selectbox("Choose the INTERVAL", config.INTERVALS)
    symbol = st.selectbox("Choose the SYMBOL", config.SYMBOLS)

    if st.button("Apply"):
        print("Apply")
        #st.session_state.interval = interval
        #st.session_state.symbol = symbol
        #st.rerun()
        
    prices, dates = [],[]
    for index,kline in df.iterrows():
        prices.append(kline['close'])
        dates.append(kline['close_time'])
    chart_data = pd.DataFrame(prices, dates)
    st.line_chart(chart_data)
