import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta # Behtar technical library
import plotly.graph_objects as go
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="Buffett AI Pro", layout="wide")

# CSS for Premium Look
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: white; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    .buy-signal { color: #2ecc71; font-weight: bold; font-size: 20px; }
    .card { background-color: #1c2128; padding: 20px; border-radius: 15px; border-left: 5px solid #2ecc71; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# 1. Warren Buffett Criteria Functions (Fundamental)
def get_fundamentals(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Intrinsic Value (Simplified: Graham Number or Low P/E)
        pe_ratio = info.get('trailingPE', 100)
        pb_ratio = info.get('priceToBook', 100)
        promoter_holding = info.get('heldPercentInstitutions', 0) * 100 # Approx for India
        
        # Criteria Check
        is_undervalued = pe_ratio < 25 and pb_ratio < 3
        is_strong_promoter = promoter_holding > 40
        
        return is_undervalued, is_strong_promoter, pe_ratio, promoter_holding
    except:
        return False, False, 0, 0

# 2. Technical Analysis (RSI, MACD, SuperTrend)
def add_technicals(df):
    # RSI
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    # MACD (New Safe Method)
    macd = ta.macd(df['Close'])
    # Yahan hum column ka naam dhundne ki jagah pehle 2 columns utha lenge
    df['MACD'] = macd.iloc[:, 0]        # MACD Line
    df['MACD_Signal'] = macd.iloc[:, 1] # Signal Line
    
    # SuperTrend
    sti = ta.supertrend(df['High'], df['Low'], df['Close'], length=7, multiplier=3)
    # SuperTrend ke liye bhi pehla column
    df['ST'] = sti.iloc[:, 0]
    
    return df


# App Header
st.title("🦸‍♂️ Buffett + Technical AI Screener")
st.write("Criteria: Low P/E, Strong Promoters + RSI > 55, MACD Cross, SuperTrend +ve")

# Sidebar
st.sidebar.header("Settings")
selected_stocks = st.sidebar.multiselect(
    "Stocks Select Karein", 
    ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ITC.NS", "INFY.NS", "TATASTEEL.NS"],
    default=["RELIANCE.NS", "HDFCBANK.NS", "ITC.NS"]
)

if st.button("Scan Market Now 🚀"):
    for stock in selected_stocks:
        with st.container():
            # Data Fetching
            data = yf.download(stock, period="60d", interval="1d")
            if data.empty: continue
            
            data = add_technicals(data)
            is_undervalued, is_strong, pe, prom = get_fundamentals(stock)
            
            # Current Values
            curr_rsi = data['RSI'].iloc[-1]
            curr_macd = data['MACD'].iloc[-1]
            curr_macd_sig = data['MACD_Signal'].iloc[-1]
            curr_price = data['Close'].iloc[-1]
            curr_st = data['ST'].iloc[-1]
            volume_gain = data['Volume'].iloc[-1] > data['Volume'].rolling(20).mean().iloc[-1]

            # FINAL SELECTION LOGIC
            # Technical Conditions
            tech_pass = curr_rsi > 55 and curr_macd > curr_macd_sig and curr_price > curr_st and volume_gain
            
            # Show Result Card
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"""
                <div class="card">
                    <h3>{stock}</h3>
                    <p>Price: ₹{curr_price:.2f}</p>
                    <p>P/E Ratio: {pe:.2f} {'✅' if is_undervalued else '❌'}</p>
                    <p>Promoters: {prom:.1f}% {'✅' if is_strong else '❌'}</p>
                    <hr>
                    <p class="buy-signal">{"🚀 STRONG BUY" if (tech_pass and is_undervalued) else "⏸️ WATCHING"}</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                # Charting
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Price'))
                fig.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0), template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Note: Fundamental data fetching takes time. NSE stocks ke liye '.NS' lagana zaroori hai.")
