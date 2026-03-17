import yfinance as yf
import pandas as pd
# from alpha_vantage.timeseries import TimeSeries

# api_key = "LBCEAZFTL1V4044M"


def load_stock_data(ticker):
  # Downloading the data through yfinance 
  data = yf.download(ticker, start="2019-01-01")['Close']

  # Selecting only columns close and volume as: 
  # 1.) Volume will tell number of units traded during a time period and
  # 2.) Close will tell the final price of an asset at the end of a time period
  data = data[['Close','Volume']]
  if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)

  data.dropna(inplace=True)
  

  return data