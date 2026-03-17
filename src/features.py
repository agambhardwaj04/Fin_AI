import numpy as np
def add_features(data):

  # Adding Features and Target 
  data["MA50"] = data["Close"].rolling(50).mean()

  data["MA200"] = data["Close"].rolling(200).mean()

  data["Return"] = data["Close"].pct_change()

  data["Volatility"] = data["Return"].rolling(20).std()

  data["Target"] = np.where(data["Close"].shift(-1) > data["Close"],1 , 0)
  
  return data

