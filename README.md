# AI-Powered Stock Market Analyzer Dashboard

An interactive stock-market analytics dashboard built with Python, Streamlit, yfinance, Plotly, and scikit-learn.

## Features
- Yahoo Finance market data for AAPL, TSLA, GOOGL, and MSFT
- Candlestick chart, moving averages (MA20/MA50), and volume
- RSI (14-period) with overbought/oversold levels
- Investment calculator with estimated shares, portfolio value, and profit/loss
- Intraday comparison for selected stocks
- Experimental Linear Regression price estimate and Random Forest directional signal
- RSI-based technical signal and sample portfolio summary

## Setup
Python 3.10+ recommended. Clone and run:

```bash
git clone https://github.com/Ksharma-09/-AI-powered-stock-market-Analyzer-dashboard.git
cd -AI-powered-stock-market-Analyzer-dashboard
python -m venv .venv
```

Windows activation: `.venv\\Scripts\\activate`  
macOS/Linux activation: `source .venv/bin/activate`

```bash
pip install -r requirements.txt
streamlit run python3.py
```

## Notes
- Internet access is required for Yahoo Finance data.
- Intraday comparison may show no data outside market hours or when the provider has no data.
- ML predictions and BUY/SELL signals are experimental and are **not financial advice or guaranteed forecasts**.

## Tech stack
Python, Streamlit, yfinance, Pandas, NumPy, Plotly, ta, scikit-learn.
