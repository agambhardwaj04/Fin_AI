# 📈 Fin AI — AI Financial Analytics Platform
An end-to-end AI-powered financial analytics platform built with Python and Streamlit. Analyzes stocks using machine learning, optimizes portfolios using Markowitz theory, and answers financial questions using Google Gemini.

---

## 🚀 Features
- **Real-time Stock Analysis** — Fetches live data via yFinance for any stock ticker
- **Interactive Price Charts** — Plotly-powered candlestick and line charts
- **ML Directional Prediction** — XGBoost model with time-series cross-validation predicts next-day price movement
- **Technical Indicators** — MA50, MA200, RSI, Bollinger Bands, Volatility, Volume analysis
- **Portfolio Optimization** — Markowitz mean-variance optimization with Sharpe Ratio maximization across AAPL, MSFT, TSLA, NVDA, GOOGL
- **Risk/Return Metrics** — Annualized return, volatility, and Sharpe Ratio
- **AI Financial Assistant** — Powered by Google Gemini 2.5 Flash for stock Q&A, buy/sell recommendations, and portfolio explanations
- **Dark Finance UI** — Bloomberg Terminal-inspired dark theme with IBM Plex Mono and Syne fonts
- **CI/CD Pipeline** — GitHub Actions automatically retrains the model every Monday with fresh market data

---

## 🛠️ Tech Stack
| Layer | Technology |
|---|---|
| Frontend | Streamlit, Plotly, Custom CSS |
| ML Model | XGBoost, Scikit-learn Pipeline |
| Data | yFinance, Pandas, NumPy |
| Optimization | SciPy (SLSQP minimizer) |
| AI Assistant | Google Gemini 2.5 Flash |
| CI/CD | GitHub Actions |
| Security | python-dotenv |

---

## 📁 Project Structure
```
Fin_AI/
├── .github/
│   └── workflows/
│       └── retrain.yml     # GitHub Actions — auto retrains model every Monday
├── src/
│   ├── app.py              # Main Streamlit app
│   ├── data_loader.py      # yFinance data fetching
│   ├── features.py         # Feature engineering & technical indicators
│   ├── model.py            # XGBoost ML pipeline (run this to generate model.pkl)
│   ├── portfolio.py        # Markowitz portfolio optimization
│   ├── AI_assistant.py     # Google Gemini integration
│   └── model.pkl           # Pretrained model (auto-updated by CI/CD)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ CI/CD Pipeline
This project uses **GitHub Actions** for automated model retraining.

```
Every Monday at midnight UTC
        ↓
GitHub spins up a free Linux machine
        ↓
Installs Python + all dependencies
        ↓
Runs model.py — downloads fresh stock data, retrains XGBoost, saves model.pkl
        ↓
Commits new model.pkl back to repo
        ↓
Streamlit Cloud detects new commit → redeploys app automatically ✅
```

You can also trigger retraining manually anytime:
```
GitHub → Actions tab → Retrain Model Weekly → Run workflow
```

---

## 🤖 ML Model Details
- **Algorithm:** XGBoost Classifier
- **Target:** Next-day directional movement (1 = up, 0 = down)
- **Validation:** TimeSeriesSplit (5 folds) — prevents data leakage
- **Features:** MA50, MA200, RSI, Bollinger Bands, Volatility, Momentum, Volume
- **Pipeline:** StandardScaler → XGBoost
- **Training Data:** AAPL, MSFT, TSLA, NVDA, GOOGL (combined for generalisation)

---

## 📊 Portfolio Optimization
Uses **Markowitz Mean-Variance Optimization** via SciPy's SLSQP minimizer:
- Maximizes Sharpe Ratio (risk-adjusted return)
- No short selling (weights bounded 0–1)
- Weights sum to 100%
- Risk-free rate: 5%

---

## 🔧 Run Locally

```bash
# Clone the repo
git clone https://github.com/agambhardwaj04/Fin_AI
cd Fin_AI

# Install dependencies
pip install -r requirements.txt

# Train the model first (generates model.pkl)
cd src
python model.py

# Run the app
streamlit run app.py
```

---

## 🌍 Indian Stock Tickers
For Indian stocks, add the exchange suffix:
| Exchange | Suffix | Example |
|---|---|---|
| NSE | `.NS` | `SBIN.NS`, `TCS.NS`, `RELIANCE.NS` |
| BSE | `.BO` | `SBIN.BO`, `TCS.BO` |

---

## 🔮 Future Improvements
- Sentiment analysis from financial news
- Deep learning models (LSTM)
- Support for crypto tickers
- Email alerts for BUY/SELL signals

---

## ⚠️ Disclaimer
This platform is for **educational and research purposes only**. Nothing here constitutes financial advice. Always consult a qualified financial advisor before making investment decisions.

---

## 👤 Author
**Agam Bhardwaj**  
[GitHub](https://github.com/agambhardwaj04)
