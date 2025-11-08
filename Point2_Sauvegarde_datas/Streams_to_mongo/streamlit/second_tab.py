import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


title = "Data from Mongo"
sidebar_name = "Data from Mongo"


def run():

    st.title(title)

    from pymongo import MongoClient
    SYMBOLS = ["btcusdt", "ethusdt", "bnbusdt", "solusdt"] 
    mongo_client = MongoClient(
        host = "127.0.0.1",
        port = 27017,
        username = "datascientest", # TODO use env
        password = "dst123"         # TODO use env
    )
    db = mongo_client['binance_klines']
    for s in SYMBOLS:
        collection = db[s]
        st.markdown(f'### {s}')
        klines = collection.find()

        prices,dates = [],[]
        for kline in klines:
            prices.append(kline['close'])
            dates.append(kline['timestamp'])

       # chart_data = pd.DataFrame(np.random.randn(20, 3), columns=list("abc"))
        chart_data = pd.DataFrame(prices, dates,columns=[s])

        st.line_chart(chart_data)

    for s in SYMBOLS:
        collection = db[s]
        st.markdown(f'### {s}')
        klines = collection.find()

        df = pd.DataFrame(list(klines))
        plt.figure()

        # "up" dataframe will store the stock_prices 
        # when the closing stock price is greater
        # than or equal to the opening stock prices
        up = df[df.close >= df.open]

        # "down" dataframe will store the stock_prices
        # when the closing stock price is
        # lesser than the opening stock prices
        down = df[df.close < df.open]

        # When the stock prices have decreased, then it
        # will be represented by blue color candlestick
        col1 = 'red'
        col2 = 'green'

        # Setting width of candlestick elements
        width = .09
        width2 = .01

        # Plotting up prices of the stock
        plt.bar(up.timestamp, up.close-up.open, width, bottom=up.open, color=col1)
        plt.bar(up.timestamp, up.high-up.close, width2, bottom=up.close, color=col1)
        plt.bar(up.timestamp, up.low-up.open, width2, bottom=up.open, color=col1)

        # Plotting down prices of the stock
        plt.bar(down.timestamp, down.close-down.open, width, bottom=down.open, color=col2)
        plt.bar(down.timestamp, down.high-down.open, width2, bottom=down.open, color=col2)
        plt.bar(down.timestamp, down.low-down.close, width2, bottom=down.close, color=col2)

        # rotating the x-axis tick labels at 30degree 
        # towards right
        plt.xticks(rotation=90, ha='right')

        # displaying candlestick chart of stock data 
        st.pyplot(plt)

    #import os
    #st.write(os.listdir('.'))
    #st.image(Image.open("streamlit_app/assets/sample-image.jpg"))
