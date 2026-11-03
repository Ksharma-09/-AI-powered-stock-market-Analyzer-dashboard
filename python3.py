import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

st.set_page_config(page_title="Live Stock Market Dashboard", page_icon="📈", layout="wide")
st.title("📈 LIVE STOCK MARKET DASHBOARD")
st.caption("Market data and model outputs are informational—not financial advice.")

STOCKS = {"AAPL":"Apple", "TSLA":"Tesla", "GOOGL":"Alphabet (Google)", "MSFT":"Microsoft"}
TICKERS = list(STOCKS)

@st.cache_data(ttl=300, show_spinner=False)
def load_history(ticker, period="6mo", interval="1d"):
    try:
        frame = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if frame is None or frame.empty:
            return pd.DataFrame()
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        frame = frame.loc[:, ~frame.columns.duplicated()]
        needed = [c for c in ["Open","High","Low","Close","Volume"] if c in frame.columns]
        frame = frame[needed].copy()
        for col in needed:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return frame.dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()

def fmt_money(value):
    return f"${value:,.2f}"

st.sidebar.header("Stock selection")
ticker = st.sidebar.selectbox("Choose a stock", TICKERS, format_func=lambda x: f"{x} — {STOCKS[x]}")
period = st.sidebar.selectbox("Historical period", ["1mo","3mo","6mo","1y","2y"], index=2)
interval = "1d"
with st.spinner("Fetching market data…"):
    data = load_history(ticker, period, interval)

if data.empty or "Close" not in data:
    st.error("Market data abhi available nahi hai. Thodi der baad refresh karke try karo.")
    st.stop()

close = data["Close"].dropna()
if close.empty:
    st.error("Valid closing-price data nahi mila.")
    st.stop()
price = float(close.iloc[-1])
data["MA20"] = data["Close"].rolling(20).mean()
data["MA50"] = data["Close"].rolling(50).mean()
delta = data["Close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / loss.replace(0, np.nan)
data["RSI"] = 100 - (100 / (1 + rs))
data["RSI"] = data["RSI"].fillna(50)

st.caption(f"Selected: {STOCKS[ticker]} ({ticker}) · Latest available close: {data.index[-1].strftime('%d %b %Y')}")
k1,k2,k3,k4=st.columns(4)
k1.metric("Latest Close",fmt_money(price))
k2.metric("Period High",fmt_money(float(data["High"].max())))
k3.metric("Period Low",fmt_money(float(data["Low"].min())))
k4.metric("Latest Volume",f"{int(data['Volume'].iloc[-1]):,}" if "Volume" in data else "N/A")

st.sidebar.subheader("Investment calculator")
investment=st.sidebar.number_input("Investment amount ($)",min_value=0.0,value=10000.0,step=500.0)
buy_price=st.sidebar.number_input("Buy price per share ($)",min_value=0.01,value=100.0,step=1.0)
shares=investment/buy_price
value=shares*price
profit=value-investment
st.sidebar.metric("Estimated current value",fmt_money(value))
st.sidebar.metric("Unrealized P/L",fmt_money(profit),delta=fmt_money(profit))

c1,c2=st.columns(2)
with c1:
    fig=go.Figure()
    fig.add_trace(go.Candlestick(x=data.index,open=data["Open"],high=data["High"],low=data["Low"],close=data["Close"],name="Price"))
    fig.add_trace(go.Scatter(x=data.index,y=data["MA20"],name="MA 20"))
    fig.add_trace(go.Scatter(x=data.index,y=data["MA50"],name="MA 50"))
    fig.update_layout(template="plotly_dark",height=430,title=f"{ticker} — Price & Moving Averages",xaxis_rangeslider_visible=False)
    st.plotly_chart(fig,use_container_width=True)
with c2:
    rfig=go.Figure(go.Scatter(x=data.index,y=data["RSI"],name="RSI"))
    rfig.add_hline(y=70,line_dash="dash",line_color="red")
    rfig.add_hline(y=30,line_dash="dash",line_color="green")
    rfig.update_layout(template="plotly_dark",height=430,title="Relative Strength Index (14)",yaxis_range=[0,100])
    st.plotly_chart(rfig,use_container_width=True)

st.subheader("🤖 Model insights")
clean=data.dropna(subset=["Close"]).copy()
if len(clean)>=10:
    X=clean[["Close"]].values
    y=clean["Close"].shift(-1).dropna().values
    X=X[:-1]
    lr=LinearRegression().fit(X,y)
    predicted=float(lr.predict([[price]])[0])
    pct=(predicted-price)/price*100 if price else 0
    a,b=st.columns(2)
    a.metric("Linear regression next-step estimate",fmt_money(predicted),f"{pct:+.2f}%")
    st.caption("A simple historical regression baseline. It is not a reliable forecast of future market prices.")
else:
    st.warning("Not enough observations to train a regression model.")

# Classifier uses lagged technical features and next-period direction; no future data in features.
ml= data.copy()
ml["Next_Close"]=ml["Close"].shift(-1)
ml["Target"]=(ml["Next_Close"]>ml["Close"]).astype(int)
ml=ml.dropna(subset=["RSI","MA20","MA50","Volume","Next_Close"])
features=["RSI","MA20","MA50","Volume"]
if len(ml)>=40 and ml["Target"].nunique()==2:
    split=max(30,int(len(ml)*0.8))
    train=ml.iloc[:split]
    test=ml.iloc[split:]
    clf=RandomForestClassifier(n_estimators=150,max_depth=5,random_state=42,class_weight="balanced")
    clf.fit(train[features],train["Target"])
    accuracy=clf.score(test[features],test["Target"]) if len(test)>0 else np.nan
    signal=int(clf.predict(ml[features].tail(1))[0])
    x,y=st.columns(2)
    x.metric("Random Forest direction signal","UP / bullish" if signal else "DOWN / bearish")
    y.metric("Chronological holdout accuracy",f"{accuracy:.1%}" if pd.notna(accuracy) else "N/A")
    st.caption("Classifier accuracy is historical holdout performance and may change. Directional signal is not a recommendation.")
else:
    st.info("Directional classifier needs more history and both target classes; currently skipped.")

rsi=float(data["RSI"].iloc[-1])
st.subheader("Technical indicator summary")
if rsi>=70: st.warning(f"RSI {rsi:.1f}: potentially overbought zone")
elif rsi<=30: st.success(f"RSI {rsi:.1f}: potentially oversold zone")
else: st.info(f"RSI {rsi:.1f}: neutral range")

st.subheader("Compare stocks")
selected=st.multiselect("Choose symbols to compare",TICKERS,default=[ticker,"MSFT"] if ticker!="MSFT" else ["AAPL","MSFT"])
if selected:
    figc=go.Figure()
    with st.spinner("Loading comparison series…"):
        for symbol in selected:
            hist=load_history(symbol,period,"1d")
            if hist.empty:
                st.warning(f"No data for {symbol} in this period.")
                continue
            base=float(hist["Close"].dropna().iloc[0])
            if base:
                figc.add_trace(go.Scatter(x=hist.index,y=hist["Close"]/base*100,mode="lines",name=symbol))
    figc.update_layout(template="plotly_dark",title="Normalized performance (first close = 100)",yaxis_title="Indexed price",height=400)
    if figc.data: st.plotly_chart(figc,use_container_width=True)
    else: st.info("No comparison data to plot.")

st.subheader("Sample portfolio (editable in source)")
portfolio=pd.DataFrame({"Ticker":["AAPL","TSLA","GOOGL"],"Quantity":[10,5,6],"Cost basis ($)":[170,240,130]})
portfolio["Cost value"]=portfolio["Quantity"]*portfolio["Cost basis ($)"]
st.dataframe(portfolio,use_container_width=True,hide_index=True)
st.caption(f"Dashboard refreshed at {datetime.now().strftime('%H:%M:%S')} local time.")
