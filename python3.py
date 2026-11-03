import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go 
from ta.momentum import RSIIndicator
from streamlit_autorefresh import st_autorefresh
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime


#================================
#         page setup
#================================
st.set_page_config(page_title="stock Market Dashboard",layout="wide")
st_autorefresh(interval=60000, key="datarefresh")

time = datetime.now().strftime("%H:%M:%S")

title_col, live_col, time_col = st.columns([6,1,1])

with title_col:

    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px">

    <div style="display:flex;gap:3px">
        <div style="width:6px;height:20px;background:#22d3ee;border-radius:2px"></div>
        <div style="width:6px;height:26px;background:#4ade80;border-radius:2px"></div>
        <div style="width:6px;height:16px;background:#60a5fa;border-radius:2px"></div>
    </div>

    <h1 style="
    color:white;
    font-size:40px;
    font-weight:800;
    margin:0;
    text-shadow:
    0 0 6px #22d3ee,
    0 0 12px #22d3ee,
    0 0 20px #22d3ee;">
    LIVE STOCK MARKET DASHBOARD
    </h1>

    </div>

    <p style="
    color:#94a3b8;
    font-size:16px;
    margin-top:6px;">
    Real-time Data • 
    <span style="color:#4ade80;font-weight:600;">
    AI Powered Analytics
    </span>
    </p>
    """, unsafe_allow_html=True)


with live_col:

    st.markdown("""
    <div style="
    background:#064e3b;
    color:#4ade80;
    padding:6px 16px;
    border-radius:20px;
    text-align:center;
    font-weight:bold;
    box-shadow:0 0 8px #4ade80;">
    ● Live
    </div>
    """, unsafe_allow_html=True)


with time_col:

    st.markdown(f"""
    <div style="
    background:#1e3a8a;
    color:white;
    padding:6px 16px;
    border-radius:8px;
    text-align:center;
    font-weight:600;">
    {time}
    </div>
    """, unsafe_allow_html=True)
    

#================================
#         side bar
#================================
st.sidebar.header("stock selection")
stocks={
    "AAPL":"Apple Inc.",
    "TSLA":"Tesla",
    "GOOGL":"Google",
    "MSFT":"Microsoft"
}

ticker=st.sidebar.selectbox("select stock",list(stocks.keys()),format_func=lambda x:f"{x}-{stocks[x]}")
#================================
#         side bar
#================================

data=yf.download(ticker, period="6mo", interval="1h", progress=False, auto_adjust=False)
if isinstance(data.columns,pd.MultiIndex):
    data.columns=data.columns.get_level_values(0)
if data.empty:
    st.error("No Stock Data Found")
    st.stop()
    
current_price=float(data["Close"].dropna().iloc[-1])

#================================
#         INVESTMENT CALCULATOR
#================================
st.sidebar.subheader("Investment Calculator")
investment=st.sidebar.number_input("Investment",min_value=0.0,value=10000.0)
buy_price=st.sidebar.number_input("Buy Price",min_value=0.0,value=100.0)
shares=investment/buy_price if buy_price>0 else 0
portfolio_value=shares*current_price
profit=portfolio_value-investment
st.sidebar.metric("current price",round(current_price,2))
st.sidebar.metric("portfolio_value",round(portfolio_value,2))
if profit >0:
    st.sidebar.success(f"profit:{round(profit,2)}")
elif profit<0:
    st.sidebar.error(f"Loss:{round(abs(profit),2)}")
else:
    st.sidebar.info("No profit/Loss")
    
#================================
#         compare stock
#================================
st.sidebar.subheader("Compare Stock")
Compare=st.sidebar.multiselect(
    "Select Stocks",["AAPL","TSLA","GOOGL","MSFT"]
)

#================================
#         RSI INDICATOR
#================================
rsi_indicator=RSIIndicator(close=data["Close"].squeeze(), window=14)
data["RSI"]=rsi_indicator.rsi()
#================================
#         kpi card
#================================
col1,col2,col3,col4=st.columns(4)
col1.metric("current price",round(data["Close"].iloc[-1],2))
col2.metric("Period High",round(data["High"].max(),2))
col3.metric("Period Low",round(data["Low"].min(),2))
col4.metric("Volume",int(data["Volume"].iloc[-1]))

#================================
#         price chart + rsi indicator
#================================
data["MA20"]=data["Close"].rolling(20).mean()
data["MA50"]=data["Close"].rolling(50).mean()
col1 , col2=st.columns([2,2])
with col1:
  

   fig = go.Figure()

# ==============================
# Candlestick
# ==============================

   fig.add_trace(go.Candlestick(
       x=data.index,
       open=data["Open"],
       high=data["High"],
       low=data["Low"],
       close=data["Close"],
       name="Price",
       increasing_line_color="#00ff9f",
       decreasing_line_color="#ff4d6d",
       increasing_fillcolor="#00ff9f",
      decreasing_fillcolor="#ff4d6d"
    ))

# ==============================
# Glow effect PRICE
# ==============================

   fig.add_trace(go.Scatter(
      x=data.index,
      y=data["Close"],
      line=dict(color="rgba(56,189,248,0.2)", width=12),
      showlegend=False
    ))

   fig.add_trace(go.Scatter(
      x=data.index,
      y=data["Close"],
      line=dict(color="rgba(56,189,248,0.4)", width=8),
      showlegend=False
    ))

   fig.add_trace(go.Scatter(
      x=data.index,
      y=data["Close"],
      line=dict(color="#38bdf8", width=3),
      name="Price Line"
    ))

# ==============================
# MA20 Glow
# ==============================

   fig.add_trace(go.Scatter(
      x=data.index,
      y=data["MA20"],
      line=dict(color="rgba(34,197,94,0.2)", width=10),
      showlegend=False
    ))

   fig.add_trace(go.Scatter(
       x=data.index,
       y=data["MA20"],
       line=dict(color="#22c55e", width=3),
       name="MA20"
     ))

# ==============================
# MA50 Glow
# ==============================

   fig.add_trace(go.Scatter(
      x=data.index,
      y=data["MA50"],
      line=dict(color="rgba(250,204,21,0.2)", width=10),
      showlegend=False
    ))

   fig.add_trace(go.Scatter(
       x=data.index,
       y=data["MA50"],
       line=dict(color="#facc15", width=3),
       name="MA50"
    ))

# ==============================
# Volume bars
# ==============================

   fig.add_trace(go.Bar(
        x=data.index,
        y=data["Volume"],
        name="Volume",
        marker_color="rgba(0,255,150,0.5)",
        yaxis="y2"
    ))

# ==============================
# Layout (dark neon theme)
# ==============================

   fig.update_layout(

       template="plotly_dark",
       height=420,

       plot_bgcolor="#020617",
       paper_bgcolor="#020617",

       font=dict(color="#94a3b8"),

          xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)"
            ),
 
        yaxis=dict(
           showgrid=True,
           gridcolor="rgba(255,255,255,0.05)"
          ),

        yaxis2=dict(
            overlaying="y",
            side="right",
            showgrid=False
         ),

        legend=dict(
             orientation="h",
             yanchor="bottom",
             y=1.02,
             xanchor="right",
            x=1
            )
        ) 

   st.plotly_chart(fig, use_container_width=True)
with col2:
    #st.subheader("RSI indicator")
    rsi_fig=go.Figure()
    rsi_fig.add_trace(go.Scatter(
        x=data.index,
        y=data["RSI"],
        line=dict(color="#c084fc",width=2),
        name="RSI"))
    rsi_fig.add_hline(y=70,line_dash="dash",line_color="red",annotation_text="Overbought",annotation_position="top right")
    rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold", annotation_position="top left")
    rsi_fig.update_layout(template="plotly_dark",height=420,margin=dict(l=10,r=10,t=40,b=10))
    st.plotly_chart(rsi_fig,use_container_width=True)
#================================
#         ai prediction
#================================  
data["Target"]=data["Close"].shift(-1)
data.dropna(inplace=True)
x=np.array(data["Close"]).reshape(-1,1)
y=np.array(data["Target"])
model=LinearRegression()
model.fit(x,y)
latest_price=data["Close"].iloc[-1]
predicted_price=model.predict([[latest_price]])[0]
ai_col,ai_col2=st.columns(2)
color="#38bdf8"
with ai_col:
    st.markdown(f"""
    <div style="
    background:#020617;
    padding:25px;
    border-radius:16px;
    border:1px solid #1f2937;
    box-shadow:0 0 25px {color};
    text-align:center;
    margin-top:20px;
    ">

    <h4 style="color:#94a3b8;">🤖 AI Price Prediction</h4>

    <h1 style="color:{color};">
    ${predicted_price:.2f}
    </h1>

    <p style="color:#9ca3af;">
    Next prediction price
    </p>

    </div>
    """, unsafe_allow_html=True)         
with ai_col2:
    #data["Future"]=data["Close"].shift(-1)
    #data["Target"]=(data["Future"]>data["Close"]).astype(int)
    data_ai=data.copy()
    data_ai["Future"]=data_ai["Close"].shift(-1)
    data_ai["Target"]=(data_ai["Future"]>data_ai["Close"]).astype(int)
    data_ai=data_ai.dropna()
    if len(data_ai)>20:
        features=data_ai[["RSI","MA20","MA50","Volume"]]
        target=data_ai["Target"]
        model=RandomForestClassifier(n_estimators=200)
        model.fit(features,target)
        latest_data=features.iloc[-1].values.reshape(1,-1)
        prediction=model.predict(latest_data)[0]
    else:
        prediction=1
    if prediction==1:
        signal="BUY SIGNAL"
        color="#22c55e"
    else:
        signal="SELL SIGNAL"
        color="#ef4444"
    st.markdown(f"""<div style="background:#020617;
                padding:25px;
                border-radius:16px;
                border:1px solid#1f2937;
                box-shadow:0 0 25px{color};
                text-align:center;
                ">
                <h4 style="color:#94a3b8;">🤖AI Trading Signal</h4>
                <h1 style="color:{color};">{signal}</h1>
                <p style="color:#9ca3af;">
                RandomForest Model</p>
                </div>""",unsafe_allow_html=True)
#================================
#         trading signal
#================================    
st.subheader("Trading Signal")
latest_rsi=data["RSI"].iloc[-1]

if latest_rsi>70:
    st.error("SELL SIGNAL(OVERBOUGHT)")
elif latest_rsi<30:
       st.success("BUY SIGNAL (OVERSOLD)")
else:
      st.info("Neutral Market")

# ==============================
# Stock Comparison (Today)
# ==============================

if Compare:

    st.subheader("📊 Stock Comparison (Today)")

    fig_Compare = go.Figure()

    colors = ["#00FFFF", "#00FF7F", "#FF4B4B", "#FFA500", "#FF00FF"]

    for i, stock in enumerate(Compare):

        # download data
        df = yf.download(stock, period="1d", interval="1m", progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or "Close" not in df.columns:
            st.warning(f"No intraday data available for {stock}.")
            continue

        # cleaning
        df = df.dropna()
        df = df.round(2)

        # datetime index
        df.index = pd.to_datetime(df.index)

        # add line
        fig_Compare.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Close"],
                mode="lines",
                name=stock,
                line=dict(width=3, color=colors[i % len(colors)]),
                marker=dict(size=10)
            ))
        

    fig_Compare.update_layout(
        template="plotly_dark",
        title="Stock Comparison Today",
        xaxis_title="Time",
        yaxis_title="Price",
        height=450,
        font=dict(size=28,color="white"),
        #legend_title="Stocks",
        xaxis=dict(
            title_font=dict(size=22),
            tickfont=dict(size=18)),
        yaxis=dict(title_font=dict(size=22),tickfont=dict(size=18)),
        hoverlabel=dict(font_size=20,font_family="Arial Black"))

    if fig_Compare.data:
        st.plotly_chart(fig_Compare, use_container_width=True)
    else:
        st.info("No comparison data is currently available.")
#================================
#         portfolio summary
#================================

portfolio=pd.DataFrame({
    "stock":["AAPL","TSLA","GOOGL"],
    "Quantity":[10,5,6],
    "Price":[170,240,130]  
     })


portfolio["Total Value"]=portfolio["Quantity"]*portfolio["Price"]
styled_portfolio = portfolio.style \
.set_properties(**{
    'background-color':'#0e1a2b',
    'color':'white',
    'border-color':'#00FFFF',
    'font-size':'16px'
}) \
.set_table_styles([
{
'selector':'th',
'props':[
('background-color','#001f3f'),
('color','#00FFFF'),
('font-size','18px'),
('text-align','center')
]
},
{
'selector':'td',
'props':[('text-align','center')]
}
])

st.dataframe(styled_portfolio)