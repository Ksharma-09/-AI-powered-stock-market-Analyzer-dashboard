# AI-Powered Stock Market Analyzer Dashboard
An interactive AI-powered Stock Market Analytics Dashboard built with Python, Streamlit, and Machine Learning. Features live market data visualization, technical indicators (MA & RSI), Random Forest-based trend prediction, stock comparison, portfolio analysis, and historical performance evaluation using annualized returns, volatility, Sharpe ratio, and maximum drawdown.

## Features
- Historical OHLC price chart with 20/50-day moving averages
- RSI (14-period) and overbought/oversold reference levels
- Latest close, period high/low, and volume KPIs
- Investment value and unrealized P/L calculator
- Linear-regression next-step baseline (educational, not investment advice)
- Random Forest directional signal with chronological holdout accuracy
- Normalized stock comparison chart
- Sample portfolio table

## Run locally
Python 3.10 or 3.11 recommended.

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run python3.py
```

## Deploy on Streamlit Community Cloud
1. Push this repository to GitHub.
2. Open https://share.streamlit.io/ and sign in.
3. Create app → choose this repository and branch `main`.
4. Set **Main file path** to `python3.py`.
5. Deploy. Dependencies are read from `requirements.txt`.

## Notes / limitations
- Yahoo Finance data availability, delay, intervals, and coverage can vary.
- The app handles empty/failed data responses with a user-facing message.
- ML outputs are educational demonstrations, not reliable price forecasts or buy/sell advice.
- Past performance and holdout accuracy do not guarantee future results.
- The portfolio shown is sample data; edit the values in source before using it.
