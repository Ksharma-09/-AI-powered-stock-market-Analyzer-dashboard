import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

st.set_page_config(page_title="Live Stock Market Dashboard", page_icon="📈", layout="wide")

# Consistent dark/neon visual theme; safe fallback to Streamlit defaults.
st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background: radial-gradient(ellipse at top left, #13243b 0%, #0b1220 42%, #080d16 100%); color: #e8f0ff; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #101d30, #0b1220); border-right: 1px solid #243b59; }
[data-testid="stMetric"] { background: rgba(20, 35, 55, .72); border: 1px solid #263e5d; padding: 14px; border-radius: 12px; }
h1, h2, h3 { color: #eaf2ff; }
.stCaption, [data-testid="stCaptionContainer"] { color: #9fb4d1; }
div[data-testid="stPlotlyChart"] { border: 1px solid #203753; border-radius: 12px; padding: 5px; background: rgba(10, 18, 31, .5); }
</style>
""", unsafe_allow_html=True)

st.title("📈 LIVE STOCK MARKET DASHBOARD")
st.caption("Market data and model outputs are informational—not financial advice.")

STOCKS = {"AAPL": "Apple", "TSLA": "Tesla", "GOOGL": "Alphabet (Google)", "MSFT": "Microsoft"}
TICKERS = list(STOCKS)
PLOT_TEMPLATE = "plotly_dark"

@st.cache_data(ttl=300, show_spinner=False)
def load_history(ticker, period="6mo", interval="1d"):
    """Download and normalize Yahoo Finance data, including MultiIndex responses."""
    try:
        frame = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if frame is None or frame.empty:
            return pd.DataFrame()
        if isinstance(frame.columns, pd.MultiIndex):
            # yfinance may return (Price, Ticker) or (Ticker, Price).
            price_level = next((i for i, level in enumerate(frame.columns.names)
                                if level and str(level).lower() in {"price", "prices"}), 0)
            frame.columns = frame.columns.get_level_values(price_level)
        frame = frame.loc[:, ~frame.columns.duplicated()].copy()
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if "Close" not in frame.columns:
            return pd.DataFrame()
        return frame.dropna(subset=["Close"]).sort_index()
    except Exception as exc:
        # Keep the app usable when Yahoo Finance is temporarily unavailable.
        return pd.DataFrame()

def fmt_money(value):
    return f"${value:,.2f}" if pd.notna(value) else "N/A"

def last_valid(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.iloc[-1]) if not values.empty else np.nan

def safe_rsi(close, window=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = gain.div(loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    # Flat prices => neutral; zero losses but positive gains => 100.
    rsi = rsi.mask((loss == 0) & (gain > 0), 100)
    rsi = rsi.mask((gain == 0) & (loss > 0), 0)
    rsi = rsi.mask((gain == 0) & (loss == 0), 50)
    return rsi.clip(0, 100)

st.sidebar.header("Stock selection")
ticker = st.sidebar.selectbox("Choose a stock", TICKERS, format_func=lambda x: f"{x} — {STOCKS[x]}")
period = st.sidebar.selectbox("Historical period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)

with st.spinner("Fetching market data…"):
    data = load_history(ticker, period, "1d")

if data.empty or "Close" not in data.columns:
    st.error("Market data abhi available nahi hai. Thodi der baad refresh karke try karo.")
    st.info("Yahoo Finance connection check karo ya kuch der baad refresh karo.")
    st.stop()

close = data["Close"].dropna()
if close.empty:
    st.error("Valid closing-price data nahi mila.")
    st.stop()

price = float(close.iloc[-1])
data["MA20"] = close.reindex(data.index).rolling(20, min_periods=20).mean()
data["MA50"] = close.reindex(data.index).rolling(50, min_periods=50).mean()
data["RSI"] = safe_rsi(close)

latest_date = data.index[-1]
date_text = latest_date.strftime("%d %b %Y") if hasattr(latest_date, "strftime") else str(latest_date)
st.caption(f"Selected: {STOCKS[ticker]} ({ticker}) · Latest available close: {date_text}")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Latest Close", fmt_money(price))
high = last_valid(data["High"]) if "High" in data else np.nan
low = last_valid(data["Low"]) if "Low" in data else np.nan
volume = last_valid(data["Volume"]) if "Volume" in data else np.nan
k2.metric("Period High", fmt_money(pd.to_numeric(data["High"], errors="coerce").max()) if "High" in data else "N/A")
k3.metric("Period Low", fmt_money(pd.to_numeric(data["Low"], errors="coerce").min()) if "Low" in data else "N/A")
k4.metric("Latest Volume", f"{int(volume):,}" if pd.notna(volume) else "N/A")

st.sidebar.subheader("Investment calculator")
investment = st.sidebar.number_input("Investment amount ($)", min_value=0.0, value=10000.0, step=500.0)
buy_price = st.sidebar.number_input("Buy price per share ($)", min_value=0.01, value=100.0, step=1.0)
shares = investment / buy_price
current_value = shares * price
profit = current_value - investment
st.sidebar.metric("Estimated current value", fmt_money(current_value))
st.sidebar.metric("Unrealized P/L", fmt_money(profit), delta=fmt_money(profit))

# Candlestick chart gracefully degrades to a close-price line if OHLC is incomplete.
left, right = st.columns(2)
with left:
    fig = go.Figure()
    if all(col in data.columns for col in ["Open", "High", "Low", "Close"]):
        ohlc = data.dropna(subset=["Open", "High", "Low", "Close"])
        if not ohlc.empty:
            fig.add_trace(go.Candlestick(x=ohlc.index, open=ohlc["Open"], high=ohlc["High"],
                                       low=ohlc["Low"], close=ohlc["Close"], name="Price"))
        else:
            fig.add_trace(go.Scatter(x=close.index, y=close, mode="lines", name="Close"))
    else:
        fig.add_trace(go.Scatter(x=close.index, y=close, mode="lines", name="Close"))
    fig.add_trace(go.Scatter(x=data.index, y=data["MA20"], name="MA 20", line={"width": 2}))
    fig.add_trace(go.Scatter(x=data.index, y=data["MA50"], name="MA 50", line={"width": 2}))
    fig.update_layout(template=PLOT_TEMPLATE, height=430, title=f"{ticker} — Price & Moving Averages",
                      xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=55, b=20))
    st.plotly_chart(fig, use_container_width=True)
with right:
    rfig = go.Figure(go.Scatter(x=data.index, y=data["RSI"], name="RSI", line={"width": 2}))
    rfig.add_hline(y=70, line_dash="dash", line_color="#ff5c7a")
    rfig.add_hline(y=30, line_dash="dash", line_color="#38d9a9")
    rfig.update_layout(template=PLOT_TEMPLATE, height=430, title="Relative Strength Index (14)",
                      yaxis_range=[0, 100], margin=dict(l=20, r=20, t=55, b=20))
    st.plotly_chart(rfig, use_container_width=True)

st.subheader("🤖 Model insights")
clean = data.dropna(subset=["Close"]).copy()
if len(clean) >= 10:
    X = clean[["Close"]].iloc[:-1].to_numpy()
    y_next = clean["Close"].shift(-1).dropna().to_numpy()
    if len(X) == len(y_next) and len(y_next) > 0:
        regression = LinearRegression().fit(X, y_next)
        predicted = float(regression.predict([[price]])[0])
        pct = (predicted - price) / price * 100 if price else 0
        col_a, _ = st.columns(2)
        col_a.metric("Linear regression next-step estimate", fmt_money(predicted), f"{pct:+.2f}%")
        st.caption("Simple historical regression baseline—not a reliable forecast of future prices.")
    else:
        st.info("Not enough aligned observations for regression.")
else:
    st.warning("Not enough observations to train a regression model.")

# Classifier: chronological split, lagged features, no random train/test leakage.
ml = data.copy()
ml["Next_Close"] = ml["Close"].shift(-1)
ml["Target"] = (ml["Next_Close"] > ml["Close"]).astype(int)
features = ["RSI", "MA20", "MA50", "Volume"]
ml = ml.replace([np.inf, -np.inf], np.nan).dropna(subset=features + ["Next_Close"])
if len(ml) >= 40 and ml["Target"].nunique() == 2:
    split = min(max(30, int(len(ml) * 0.8)), len(ml) - 1)
    train, test = ml.iloc[:split], ml.iloc[split:]
    if not test.empty and train["Target"].nunique() == 2:
        classifier = RandomForestClassifier(n_estimators=150, max_depth=5, random_state=42,
                                           class_weight="balanced", n_jobs=-1)
        classifier.fit(train[features], train["Target"])
        accuracy = classifier.score(test[features], test["Target"])
        latest_features = ml[features].tail(1)
        signal = int(classifier.predict(latest_features)[0])
        probabilities = classifier.predict_proba(latest_features)[0]
        classes = list(classifier.classes_)
        bullish_probability = float(probabilities[classes.index(1)]) if 1 in classes else 0.0
        bearish_probability = float(probabilities[classes.index(0)]) if 0 in classes else 0.0

        # Combine model direction with simple trend/RSI filters to avoid overclaiming.
        latest_close = last_valid(data["Close"])
        latest_ma20 = last_valid(data["MA20"])
        latest_rsi = last_valid(data["RSI"])
        if signal == 1 and latest_close > latest_ma20 and latest_rsi < 70:
            trade_signal = "BUY"
            signal_note = "Model bullish + price above MA20 + RSI below 70"
            signal_confidence = bullish_probability
        elif signal == 0 and latest_close < latest_ma20 and latest_rsi > 30:
            trade_signal = "SELL"
            signal_note = "Model bearish + price below MA20 + RSI above 30"
            signal_confidence = bearish_probability
        else:
            trade_signal = "HOLD"
            signal_note = "Model and technical filters do not align clearly"
            signal_confidence = max(bullish_probability, bearish_probability)

        st.subheader("📊 Buy / Sell / Hold signal")
        signal_color = {"BUY": "#20c997", "SELL": "#ff5c7a", "HOLD": "#f0b429"}[trade_signal]
        st.markdown(
            f'<div style="padding:18px;border-radius:14px;border:1px solid {signal_color};'
            f'background:rgba(16,29,48,.85);text-align:center">'
            f'<div style="font-size:30px;font-weight:800;color:{signal_color}">{trade_signal}</div>'
            f'<div style="font-size:16px;color:#dbe7f7">{signal_note}</div></div>',
            unsafe_allow_html=True
        )
        sig1, sig2, sig3 = st.columns(3)
        sig1.metric("Model probability (signal class)", f"{signal_confidence:.1%}")
        sig2.metric("Bullish probability", f"{bullish_probability:.1%}")
        sig3.metric("Chronological holdout accuracy", f"{accuracy:.1%}")
        st.caption("Signals are rule-based educational estimates, not financial advice or guaranteed predictions.")
    else:
        st.info("Classifier skipped: training data needs both classes and a non-empty holdout.")
else:
    st.info("Directional classifier needs more history and both target classes; currently skipped.")

rsi_value = last_valid(data["RSI"])
st.subheader("Technical indicator summary")
if pd.isna(rsi_value):
    st.info("RSI needs more price observations for this selected period.")
elif rsi_value >= 70:
    st.warning(f"RSI {rsi_value:.1f}: potentially overbought zone")
elif rsi_value <= 30:
    st.success(f"RSI {rsi_value:.1f}: potentially oversold zone")
else:
    st.info(f"RSI {rsi_value:.1f}: neutral range")

# Descriptive buy-and-hold diagnostics for the selected historical window.
returns = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
if len(returns) >= 2:
    st.subheader("📉 Historical risk & return profile")
    cumulative = (1 + returns).cumprod()
    running_peak = cumulative.cummax()
    drawdown = cumulative / running_peak - 1
    annualized_return = float(cumulative.iloc[-1] ** (252 / len(returns)) - 1) if cumulative.iloc[-1] > 0 else np.nan
    annualized_volatility = float(returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 else np.nan
    sharpe = annualized_return / annualized_volatility if annualized_volatility > 0 else np.nan
    risk1, risk2, risk3 = st.columns(3)
    risk1.metric("Annualized historical return", f"{annualized_return:.2%}" if np.isfinite(annualized_return) else "N/A")
    risk2.metric("Annualized volatility", f"{annualized_volatility:.2%}" if np.isfinite(annualized_volatility) else "N/A")
    risk3.metric("Maximum drawdown", f"{float(drawdown.min()):.2%}")
    st.metric("Historical Sharpe ratio (rf = 0)", f"{sharpe:.2f}" if np.isfinite(sharpe) else "N/A")
    st.caption("Descriptive buy-and-hold metrics for the selected period, annualized using 252 trading days. Not a forecast; Sharpe assumes zero risk-free rate.")

st.subheader("Compare stocks")
selected = st.multiselect("Choose symbols to compare", TICKERS,
                          default=[ticker, "MSFT"] if ticker != "MSFT" else ["AAPL", "MSFT"])
if selected:
    comparison = go.Figure()
    with st.spinner("Loading comparison series…"):
        for symbol in selected:
            hist = load_history(symbol, period, "1d")
            if hist.empty or "Close" not in hist.columns:
                st.warning(f"No data for {symbol} in this period.")
                continue
            series = hist["Close"].dropna()
            if series.empty:
                st.warning(f"No valid closing prices for {symbol}.")
                continue
            base = float(series.iloc[0])
            if np.isfinite(base) and base != 0:
                comparison.add_trace(go.Scatter(x=series.index, y=series / base * 100,
                                           mode="lines", name=symbol))
    comparison.update_layout(template=PLOT_TEMPLATE, title="Normalized performance (first close = 100)",
                            yaxis_title="Indexed price", height=400, margin=dict(l=20, r=20, t=55, b=20))
    if comparison.data:
        st.plotly_chart(comparison, use_container_width=True)
    else:
        st.info("No comparison data to plot for the selected symbols.")

st.subheader("Sample portfolio")
st.caption("Illustrative sample holdings only. Edit these values in the source code to customize.")
portfolio = pd.DataFrame({"Ticker": ["AAPL", "TSLA", "GOOGL"], "Quantity": [10, 5, 6],
                          "Cost basis ($)": [170, 240, 130]})
portfolio["Cost value ($)"] = portfolio["Quantity"] * portfolio["Cost basis ($)"]
st.dataframe(portfolio, use_container_width=True, hide_index=True)
st.caption(f"Dashboard refreshed at {datetime.now().strftime('%H:%M:%S')} local time.")
