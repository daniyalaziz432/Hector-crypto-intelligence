# ⬡ HECTOR — Crypto Financial Intelligence System

A Bloomberg Terminal-inspired crypto analysis dashboard built entirely on free data sources. Zero API keys required.

---

## Features

- **20 Enhancements** — ML pipeline, backtesting, live feed, paper trading, portfolio optimisation, and more
- **50 Assets** — Bitcoin, Ethereum, and 48 other cryptocurrencies 
- **100% Free Data** — yfinance · CoinGecko · alternative.me · Reddit · CryptoCompare
- **Multi-Model Ensemble** — Random Forest, Gradient Boosting, Logistic Regression, XGBoost (optional), LightGBM (optional)
- **Advanced Analytics** — Triple Barrier Labels, HRP Portfolio, Efficient Frontier, HMM Regime Detection, SHAP explainability

---

## Installation

```bash
git clone https://github.com/daniyalaziz432/Hector-crypto-intelligence


---

## click the link below to acess the app

 [huggingface.co/spaces](https://huggingface.co/spaces/daniyalaziz/hector-crypto)
   

---

## Tabs

| Tab | Description |
|---|---|
| Market Overview | Live price, indicators, regime detection |
| Signals | Triple-barrier labels and ML signals |
| Models | Ensemble model metrics, SHAP importance, ROC curves |
| Backtest | Equity curve, drawdown, walk-forward validation |
| Risk | VaR, CVaR, stress testing, rolling Sharpe |
| Portfolio | HRP weights, correlation matrix, efficient frontier |
| Research | Fractional differentiation, entropy, CUSUM |
| Sentiment | Reddit sentiment, crypto news, on-chain proxies |
| Alpha Lab | Feature stability, drift detection, A/B testing |
| Live Feed | Real-time 1-min bar polling |
| Paper Trading | Simulated order management with risk controls |
| System | Health check, library status, data sources |
| Export | CSV, labels, signals, PDF report |

---

## Data Sources (all free, no keys)

- **yfinance** — OHLCV data (~2,000 requests/hour)
- **CoinGecko** — Market data (30 requests/minute)
- **alternative.me** — Fear & Greed Index (50 requests/minute)
- **Reddit public JSON** — Sentiment analysis
- **CryptoCompare** — Crypto news feed
- **blockchain.info** — On-chain BTC stats

---

## Optional Libraries (for enhanced functionality)

```bash
pip install xgboost lightgbm optuna shap hmmlearn lime joblib pyarrow statsmodels fpdf2
```

HECTOR degrades gracefully if any optional library is missing.

---

Created by **Daniyal Aziz**
