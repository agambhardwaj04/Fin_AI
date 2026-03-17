import numpy as np
import pandas as pd
import yfinance as yf

def optimize_portfolio(stocks):
  prices = yf.download(stocks,start="2019-01-01")["Close"]

  returns = prices.pct_change().dropna()
  weights = np.random.random(len(stocks))

  weights = weights / np.sum(weights)

  portfolio_return = np.sum(returns.mean() * weights) * 252

  portfolio_risk = np.sqrt(
    np.dot(weights.T,
           np.dot(returns.cov() * 252, weights))
  )

  portfolio = pd.DataFrame({
    "stock":stocks,
    "Weight": weights
  })

  return portfolio, portfolio_return, portfolio_risk