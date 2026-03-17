import numpy as np
import pandas as pd

def add_features(data):
    data = data.copy()

    # Trend features
    data["MA50"] = data["Close"].rolling(50).mean()
    data["MA200"] = data["Close"].rolling(200).mean()
    data["MA_ratio"] = data["MA50"] / data["MA200"]
    data["Price_to_MA50"] = data["Close"] / data["MA50"]

    # Momentum features
    data["Return"] = data["Close"].pct_change()
    data["Momentum_5"] = data["Close"].pct_change(5)
    data["Momentum_10"] = data["Close"].pct_change(10)
    data["Momentum_20"] = data["Close"].pct_change(20)

    # Volatility features
    data["Volatility"] = data["Return"].rolling(20).std()
    data["Volatility_10"] = data["Return"].rolling(10).std()

    # Volume features
    data["Volume_MA20"] = data["Volume"].rolling(20).mean()
    data["Volume_ratio"] = data["Volume"] / data["Volume_MA20"]

    # RSI
    delta = data["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Band position
    rolling_mean = data["Close"].rolling(20).mean()
    rolling_std = data["Close"].rolling(20).std()
    data["BB_position"] = (data["Close"] - rolling_mean) / (2 * rolling_std)

    # Target — next day movement
    data["Target"] = np.where(data["Close"].shift(-1) > data["Close"], 1, 0)

    data.dropna(inplace=True)

    return data