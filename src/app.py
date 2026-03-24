import streamlit as st
import plotly.express as px
import pandas as pd

from data_loader import load_stock_data
from features import add_features
from model import train_model, load_model
from portfolio import optimize_portfolio
from AI_assistant import generate_ai_insight

# ── Page config 
st.set_page_config(
    page_title="FinAI Analytics",
    page_icon="📈",
    layout="wide"
)

# ── CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');

/* Force dark background always */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #080c12 !important;
    background-image: 
        linear-gradient(rgba(0, 212, 255, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0, 212, 255, 0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    color: #c9d1d9 !important;
    font-family: 'Syne', sans-serif;
}

[data-testid="stSidebar"] {
    background-color: #0d1117 !important;
    border-right: 1px solid #1c2a3a;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Title */
h1 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: clamp(1.8rem, 4vw, 2.8rem) !important;
    background: linear-gradient(90deg, #00d4ff, #00ff9d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem !important;
}

/* Subheaders */
h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    color: #e6edf3 !important;
    border-left: 3px solid #00d4ff;
    padding-left: 0.6rem;
    margin-top: 1.5rem !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0d1f2d, #0a1628);
    border: 1px solid #1c3a5e;
    border-radius: 10px;
    padding: 1rem 1.2rem !important;
    box-shadow: 0 0 20px rgba(0, 212, 255, 0.05);
    transition: box-shadow 0.3s ease;
}
[data-testid="stMetric"]:hover {
    box-shadow: 0 0 30px rgba(0, 212, 255, 0.15);
}
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    color: #58a6ff !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.5rem !important;
    color: #00ff9d !important;
}

/* Input box */
[data-testid="stTextInput"] input {
    background-color: #0d1117 !important;
    border: 1px solid #1c3a5e !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1rem !important;
    padding: 0.5rem 0.8rem !important;
    transition: border-color 0.2s;
}
[data-testid="stTextInput"] input:focus {
    border-color: #00d4ff !important;
    box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.2) !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #1c3a5e !important;
    border-radius: 10px !important;
    overflow: hidden;
}

/* Plotly chart background */
.js-plotly-plot .plotly .bg {
    fill: #080c12 !important;
}

/* Divider */
hr {
    border-color: #1c2a3a !important;
    margin: 1.5rem 0 !important;
}

/* st.write text */
p, .stMarkdown p {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
    color: #8b949e !important;
}

/* Ticker badge */
.ticker-badge {
    display: inline-block;
    background: linear-gradient(135deg, #0a2a4a, #0d1f2d);
    border: 1px solid #00d4ff44;
    border-radius: 6px;
    padding: 0.2rem 0.7rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: #00d4ff;
    letter-spacing: 0.1em;
    margin-bottom: 1rem;
}

/* Section card wrapper */
.section-card {
    background: linear-gradient(135deg, #0d1117, #0a1628);
    border: 1px solid #1c3a5e;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)

# ── Header 
st.title("AI Financial Analytics")
st.markdown("<p style='color:#58a6ff;font-family:IBM Plex Mono,monospace;font-size:0.8rem;'>Powered by XGBoost · Markowitz Optimization · yFinance</p>", unsafe_allow_html=True)
st.markdown("---")

# ── Input 
ticker = st.text_input("Enter Stock Ticker", "AAPL").upper()

if ticker:
    st.markdown(f"<div class='ticker-badge'>⬤ {ticker}</div>", unsafe_allow_html=True)

    data = load_stock_data(ticker)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    # ── Chart 
    st.subheader("Price Trend")
    fig = px.line(data, x=data.index, y="Close")
    fig.update_layout(
        paper_bgcolor="#080c12",
        plot_bgcolor="#080c12",
        font=dict(family="IBM Plex Mono", color="#8b949e"),
        xaxis=dict(gridcolor="#1c2a3a", showgrid=True),
        yaxis=dict(gridcolor="#1c2a3a", showgrid=True),
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified"
    )
    fig.update_traces(line_color="#00d4ff", line_width=1.5)
    st.plotly_chart(fig, use_container_width=True)

    # ── Model 
    st.subheader("ML Predictions")
    data = add_features(data)
    from model import load_model
    import os

    @st.cache_resource
    def get_model():
        if os.path.exists(""):
            return load_model("m.pkl")
        else:
            st.warning("model.pkl not found — training now (this may take a minute)...")
            return train_model(data)

    model, accuracy, cv_scores = get_model()


    col1, col2, col3, col4 = st.columns(4)
    returns = data["Close"].pct_change().dropna()
    avg_return = returns.mean() * 252
    volatility = returns.std() * (252 ** 0.5)

    col1.metric("Model Accuracy", f"{round(accuracy * 100, 2)}%")
    col2.metric("CV Score", f"{round(cv_scores * 100, 2)}%")
    col3.metric("Annual Return", f"{round(avg_return * 100, 2)}%")
    col4.metric("Volatility", f"{round(volatility * 100, 2)}%")
    #  ── Trading Signals
    latest_data = data.iloc[-1:][[
    "MA_ratio", "Price_to_MA50",
    "Momentum_5", "Momentum_10", "Momentum_20",
    "Volatility", "Volatility_10",
    "Volume_ratio",
    "RSI", "BB_position"
    ]]

    prediction = model.predict(latest_data)[0]
    signal = "BUY" if prediction == 1 else "SELL"

    probs = model.predict_proba(latest_data)[0]
    confidence = probs[1]

    color_map = {
    "BUY": "#00ff9d",
    "HOLD": "#ffd166",
    "SELL": "#ff4d4d"
}

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #0d1117, #0a1628);
        border: 1px solid {color_map[signal]};
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 1rem;
    ">
    <h2 style="color:{color_map[signal]}; margin:0;">
        {signal} SIGNAL
    </h2>
    <p style="color:#8b949e; font-size:0.9rem;">
        Confidence: {round(confidence*100, 2)}%
    </p>
    </div>
    """, unsafe_allow_html=True)
    if signal == "BUY":
        st.success("Momentum and indicators suggest upward trend 📈")
    elif signal == "SELL":
        st.error("Market indicators show potential downside 📉")
    else:
        st.warning("Market is neutral, wait for confirmation ⏳")

    # ── Portfolio 
    st.markdown("---")
    st.subheader("Portfolio Optimization")
    stocks = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL"]
    portfolio, portfolio_return, risk, sharpe = optimize_portfolio(stocks)

    col5, col6, col7 = st.columns(3)
    col5.metric("Expected Return", f"{round(portfolio_return * 100, 2)}%")
    col6.metric("Portfolio Risk", f"{round(risk * 100, 2)}%")
    col7.metric("Sharpe Ratio", f"{round(sharpe, 2)}")

    st.dataframe(portfolio, use_container_width=True)

# ── AI Assistant 
st.markdown("---")
st.subheader("AI Financial Assistant")
st.markdown("<p style='color:#58a6ff;font-family:IBM Plex Mono,monospace;font-size:0.75rem;'>Powered by Google Gemini · Not financial advice</p>", unsafe_allow_html=True)

question = st.text_input("Ask about the stock, predictions, portfolio, or related stocks",
                          placeholder=f"e.g. Should I buy {ticker}? What are related stocks?")

if question:
    with st.spinner("Analyzing..."):
        answer = generate_ai_insight(
            ticker=ticker,
            accuracy=accuracy,
            cv_score=cv_scores,
            avg_return=avg_return,
            volatility=volatility,
            portfolio=portfolio,
            portfolio_return=portfolio_return,
            risk=risk,
            sharpe=sharpe,
            question=question
        )

    st.markdown(f"""
    <div class='section-card'>
        <p style='color:#e6edf3;font-size:0.9rem;line-height:1.7;'>{answer}</p>
    </div>
    """, unsafe_allow_html=True)