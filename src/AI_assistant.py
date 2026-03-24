import google as genai
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or st.secrets["GEMINI_API_KEY"]

genai.configure(api_key=api_key)

def generate_ai_insight(ticker, accuracy, cv_score, avg_return, volatility, portfolio, portfolio_return, risk, sharpe, question):

    prompt = f"""
    You are a professional financial analyst AI assistant built into a financial analytics platform.
    
    Current Analysis Data:
    - Stock Ticker: {ticker}
    - ML Model Accuracy: {round(accuracy * 100, 2)}%
    - CV Score: {round(cv_score * 100, 2)}%
    - Annual Return: {round(avg_return * 100, 2)}%
    - Volatility: {round(volatility * 100, 2)}%
    
    Portfolio Optimization Results:
    - Expected Annual Return: {round(portfolio_return * 100, 2)}%
    - Portfolio Risk: {round(risk * 100, 2)}%
    - Sharpe Ratio: {round(sharpe, 2)}
    - Portfolio Weights:
    {portfolio.to_string(index=False)}
    
    Your capabilities:
    1. Answer questions about {ticker} stock
    2. Give buy/sell recommendations based on the data above
    3. Summarize the ML model predictions and what they mean
    4. Explain the portfolio weights and optimization results
    5. Suggest related stocks worth looking into alongside {ticker}
    
    Important rules:
    - Always base recommendations on the data provided above
    - Be concise but insightful — max 200 words
    - Always add a disclaimer that this is not financial advice
    - Format your response cleanly with sections if needed
    
    User Question: {question}
    """

    model = genai.GenerativeModel("gemini-2.5-flash-lite")
    response = model.generate_content(contents=prompt)

    return response.text