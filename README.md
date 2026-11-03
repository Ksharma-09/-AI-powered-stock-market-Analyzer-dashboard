# 📈 AI-Powered Stock Market Analyzer Dashboard

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest-orange)](https://scikit-learn.org/)

An interactive stock-market analytics application built with Python, Streamlit, and machine learning. Explore historical market data, technical indicators, model-based directional signals, portfolio calculations, and risk/return metrics in one dashboard.

## Key features
- Historical OHLC price visualization with 20-day and 50-day moving averages
- RSI (14-period) with reference levels
- Market summary metrics: latest close, period high/low, and volume
- Investment value and unrealized profit/loss calculator
- Educational next-step linear-regression baseline
- Random Forest directional classification with chronological holdout evaluation
- Normalized comparison of multiple stocks
- Historical buy-and-hold metrics: annualized return, volatility, Sharpe ratio (risk-free rate assumed 0), and maximum drawdown
- Sample portfolio view

## Tech stack
Python · Streamlit · yfinance · Pandas · NumPy · Plotly · scikit-learn

## Run locally
Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run python3.py
```

## Deploy to Streamlit Community Cloud
1. Sign in at https://share.streamlit.io/
2. Create a new app and select this repository, branch `main`.
3. Set **Main file path** to `python3.py`.
4. Deploy; dependencies are installed from `requirements.txt`.

## Validation
GitHub Actions installs the requirements and checks Python syntax on pushes and pull requests.

## Important limitations
- Market data is supplied by Yahoo Finance via yfinance; availability, delay, and coverage may vary.
- Model predictions and regression outputs are educational estimates—not reliable forecasts or investment advice.
- Holdout results and historical performance do not guarantee future results.
- The portfolio is illustrative sample data, not a real brokerage account.
- Risk metrics depend on the selected period and available price history.

## Disclaimer
This project is for educational and analytical purposes only. It does not provide financial advice or recommendations to buy or sell securities.
