import numpy as np
def add_features(data):
  data["MA50"] = data["Close"].rolling(50).mean()
  data["MA200"] = data["Close"].rolling(200).mean()