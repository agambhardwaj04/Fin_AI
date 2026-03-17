import streamlit as st
import plotly.express as px
import pandas as pd

from data_loader import load_stock_data
from features import add_features
from model import train_model
from portfolio import optimize_portfolio


st.title("AI Financial Analytics Platform")

ticker = st.text_input("Enter Stock Ticker", "AAPL").upper()

if ticker:
    data = load_stock_data(ticker)
    
    # Fix MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    st.subheader("Stock Price Trend")

    fig = px.line(data, x=data.index, y="Close")

    st.plotly_chart(fig)


    data = add_features(data)

    model, accuracy, cv_scores = train_model(data)

    st.write("Model Accuracy:", round(accuracy * 100, 2), "%")
    st.write("CV Score:", round(cv_scores * 100, 2), "%")



    returns = data["Close"].pct_change().dropna()

    avg_return = returns.mean() * 252

    volatility = (returns.std() * 252 ** 0.5)

    st.write("Annual Return:", avg_return * 100)

    st.write("Volatility:", volatility * 100)



    st.subheader("Portfolio Optimization")

    stocks = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL"]

    portfolio, portfolio_return, risk, sharpe = optimize_portfolio(stocks)
    st.dataframe(portfolio)
    st.write("Expected Annual Return:", portfolio_return)
    st.write("Annual Risk:", risk)
    st.write("Sharpe Ratio:", sharpe)

    # st.subheader("AI Financial Assistant")

    # question = st.text_input("Ask about the company")

    # if question:

    #     answer = generate_ai_insight(
    #         ticker,
    #         avg_return,
    #         volatility,
    #         questions
    #     )

    #     st.write(answer)