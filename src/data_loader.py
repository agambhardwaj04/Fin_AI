import yfinance as yf
import pandas as pd


def load_stock_data(ticker):
  # Downloading the data through yfinance 
  data = yf.download(ticker, start="2019-01-01")

  # Selecting only columns close and volume as: 
  # 1.) Volume will tell number of units traded during a time period and
  # 2.) Close will tell the final price of an asset at the end of a time period
  if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)
  data = data[['Close','Volume']].copy()

  data.dropna(inplace=True)
  

  return data