import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def get_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]  
    return df

def preprocess_data(df, sequence_length):
    df_pct_change = df[['Open', 'High', 'Low', 'Close']].pct_change().dropna()
    df_pct_change['Volume'] = df['Volume'].iloc[1:].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_pct_change.values)

    x, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        x.append(scaled_data[i:i + sequence_length])
        y.append(scaled_data[i + sequence_length, :4])  

    return np.array(x), np.array(y), scaler