import yfinance as yf
import pandas as pd
# from alpha_vantage.timeseries import TimeSeries

api_key = "LBCEAZFTL1V4044M"


def load_stock_data(ticker):
  # Downloading the data through yfinance 
  data = yf.download(ticker, start="2019-01-01")


  # Selecting only columns close and volume as: 
  # 1.) Volume will tell number of units traded during a time period and
  # 2.) Close will tell the final price of an asset at the end of a time period
  data = data[['Close','Volume']]

  # Downloading data from Alpha Vantage
#   data2 = TimeSeries(key=api_key, output_format='pandas')
#   data2,meta = data2.get_daily(symbol=ticker, outputsize='full')

#   data2.columns = ['Open_AV', 'High_AV', 'Low_AV', 'Close_AV', 'Volume_AV']

#   data2 = data2.sort_index()

#   data = data1.merge(
#     data2,
#     left_index=True,
#     right_index=True,
#     how="inner"
# )
  data.dropna(inplace=True)
  

  return data