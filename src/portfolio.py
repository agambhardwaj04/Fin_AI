import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

def optimize_portfolio(stocks):
    prices = yf.download(stocks, start="2019-01-01")["Close"]

    # Fix MultiIndex
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.droplevel(1)

    returns = prices.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    n = len(stocks)

    # Objective: minimize negative Sharpe Ratio
    def neg_sharpe(weights, mean_returns, cov_matrix, risk_free=0.05):
        port_return = np.sum(mean_returns * weights)
        port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (port_return - risk_free) / port_risk
        return -sharpe

    # Constraints: weights must sum to 1
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

    # Bounds: each weight between 0 and 1 (no short selling)
    bounds = tuple((0, 1) for _ in range(n))

    # Initial guess: equal weights
    initial_weights = np.array([1 / n] * n)

    # Run optimization
    result = minimize(
        neg_sharpe,
        initial_weights,
        args=(mean_returns, cov_matrix),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights = result.x

    # Calculate final portfolio metrics
    portfolio_return = np.sum(mean_returns * optimal_weights)
    portfolio_risk = np.sqrt(
        np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights))
    )
    sharpe_ratio = (portfolio_return - 0.05) / portfolio_risk

    portfolio = pd.DataFrame({
        "Stock": stocks,
        "Weight": np.round(optimal_weights * 100, 2),  # as percentage
    })
    portfolio["Weight"] = portfolio["Weight"].astype(str) + "%"

    return portfolio, round(portfolio_return, 4), round(portfolio_risk, 4), round(sharpe_ratio, 4)