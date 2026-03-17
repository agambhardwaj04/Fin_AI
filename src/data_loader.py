import yfinance as yf
import pandas as pd

def load_stock_data(ticker):
  data = yf.download(ticker, start="2019-01-01")
  data = data[['Close','Volume']]
  data.dropna(inplace=True)
  return data