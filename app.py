import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import feedparser
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- SETUP ---
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

st.set_page_config(page_title="Ultimate Pro Trader", page_icon="🚀", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #000000; color: #ffffff; }
    .box { background-color: #111111; padding: 15px; border-radius: 10px; border: 1px solid #333; margin-bottom: 10px; }
    .badge-bull { background-color: #004d00; color: #00ff00; padding: 3px 8px; border-radius: 5px; font-weight: bold; }
    .badge-bear { background-color: #4d0000; color: #ff0000; padding: 3px 8px; border-radius: 5px; font-weight: bold; }
    .badge-wait { background-color: #333; color: #aaa; padding: 3px 8px; border-radius: 5px; }
    .news-title { color: #58a6ff; font-weight: bold; text-decoration: none; font-size: 16px; }
    .impact-msg { color: #d2a8ff; font-size: 13px; margin-top: 5px; border-left: 3px solid #d2a8ff; padding-left: 8px; }
    .silver-box { border: 2px solid #C0C0C0; background-color: #1a1a1a; }
</style>
""", unsafe_allow_html=True)

# FIX: was '#silver' which is invalid CSS; changed to actual silver hex color #C0C0C0

st.title("🚀 ULTIMATE TRADER: Scanner + News AI")

# --- PART 1: DATA CONFIGURATION ---
ASSETS = {
    "🇮🇳 INDICES": {"NIFTY 50": "^NSEI", "BANK NIFTY": "^NSEBANK"},
    "🪙 CRYPTO": {"BITCOIN": "BTC-USD", "ETHEREUM": "ETH-USD", "SOLANA": "SOL-USD", "XRP": "XRP-USD"},
    "⛏️ COMMODITIES": {"SILVER (FUTURES)": "SI=F", "GOLD": "GC=F", "CRUDE OIL": "CL=F", "COPPER": "HG=F"}
}

ALL_TICKERS = []
for cat in ASSETS.values():
    ALL_TICKERS.extend(cat.values())

# --- PART 2: FUNCTIONS ---

@st.cache_data(ttl=60)
def fetch_data(tickers, period, interval):
    tickers_str = " ".join(tickers)
    data = yf.download(tickers_str, period=period, interval=interval, group_by='ticker', threads=True, progress=False)
    return data

def get_signal(df, ticker_name=""):
    if df.empty or len(df) < 50:
        return None

    df = df.copy()

    # Indicators
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    atr_series = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['ATR'] = atr_series

    # Supertrend
    st_data = ta.supertrend(df['High'], df['Low'], df['Close'], length=10, multiplier=3)
    if st_data is None or st_data.empty:
        return None
    st_dir_col = [c for c in st_data.columns if "SUPERTd_" in c]
    if not st_dir_col:
        return None
    df['Trend'] = st_data[st_dir_col[0]]

    # ADX
    adx_data = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    if adx_data is not None and not adx_data.empty:
        adx_col = [c for c in adx_data.columns if c.startswith("ADX_")]
        current_adx = adx_data[adx_col[0]].iloc[-1] if adx_col else 0
    else:
        current_adx = 0

    # Z-Score
    df['Z_Score'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()

    # Check for NaN in key columns
    if df[['Close', 'EMA_20', 'EMA_50', 'RSI', 'ATR', 'Trend', 'Z_Score']].iloc[-1].isna().any():
        return None

    close = df['Close'].iloc[-1]
    trend = df['Trend'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    ema20 = df['EMA_20'].iloc[-1]
    z_score = df['Z_Score'].iloc[-1]

    support = df['Low'].tail(20).min()
    resistance = df['High'].tail(20).max()

    action = "WAIT"
    color = "grey"
    sl = 0
    tgt = 0

    if "SILVER" in ticker_name.upper():
        if trend == 1 and close > ema20:
            action = "BUY SILVER (STRONG) 🚀" if current_adx > 20 else "BUY SILVER (WEAK) ↗️"
            color = "#00ff00" if current_adx > 20 else "#90ee90"
            sl = close - (atr * 2)
            tgt = close + (atr * 4)
        elif trend == -1 and close < ema20:
            action = "SELL SILVER (STRONG) 🩸" if current_adx > 20 else "SELL SILVER (WEAK) ↘️"
            color = "#ff0000" if current_adx > 20 else "#ff7f7f"
            sl = close + (atr * 2)
            tgt = close - (atr * 4)
    else:
        if trend == 1 and close > df['EMA_50'].iloc[-1] and rsi > 50:
            action = "BUY / LONG 🚀"
            color = "#00ff00"
            sl = close - (atr * 1.5)
            tgt = close + (atr * 3)
        elif trend == -1 and close < df['EMA_50'].iloc[-1] and rsi < 50:
            action = "SELL / SHORT 🩸"
            color = "#ff0000"
            sl = close + (atr * 1.5)
            tgt = close - (atr * 3)

    strength_txt = "Weak"
    if current_adx > 25:
        strength_txt = "Strong"
    if current_adx > 50:
        strength_txt = "Explosive 🔥"

    return {
        "action": action,
        "color": color,
        "price": close,
        "sl": sl,
        "tgt": tgt,
        "supp": support,
        "res": resistance,
        "strength": strength_txt,
        "adx": current_adx,
        "z_score": z_score
    }

# --- NEWS FUNCTIONS ---
RSS_FEEDS = {
    "General": "https://finance.yahoo.com/news/rssindex",
    "Crypto": "https://cointelegraph.com/rss",
    "India": "https://www.moneycontrol.com/rss/economy.xml"
}

def analyze_news_sentiment(text):
    score = sia.polarity_scores(text)['compound']
    if score > 0.05:
        return "BULLISH", "badge-bull"
    elif score < -0.05:
        return "BEARISH", "badge-bear"
    return "NEUTRAL", "badge-wait"

def get_impact(text):
    text = text.lower()
    if "inflation" in text or "rate" in text:
        return "⚠️ Impact: BANKNIFTY & LOANS"
    if "oil" in text:
        return "⚠️ Impact: PAINTS & TYRES"
    if "bitcoin" in text or "crypto" in text:
        return "⚠️ Impact: CRYPTO MARKET"
    if "gold" in text or "silver" in text:
        return "⚠️ Impact: PRECIOUS METALS"
    return ""

# --- PART 3: MAIN APP UI ---

tab1, tab2 = st.tabs(["📊 MARKET SCANNER", "🌍 NEWS & SENTIMENT"])

# === TAB 1: SCANNER ===
with tab1:
    st.header("📈 Live Signals & Trend Strength")

    col1, col2 = st.columns([3, 1])
    with col1:
        timeframe = st.selectbox("Timeframe Select Karo:", ["15m", "1h", "4h", "1d"])
    with col2:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

    tf_map = {"15m": "5d", "1h": "1mo", "4h": "1mo", "1d": "1y"}

    with st.spinner("Market scan aur Trend Calculate ho raha hai..."):
        raw_data = fetch_data(ALL_TICKERS, period=tf_map[timeframe], interval=timeframe)

    if raw_data is not None:
        for cat_name, tickers in ASSETS.items():
            st.subheader(cat_name)
            cols = st.columns(3)
            idx = 0
            for name, symbol in tickers.items():
                try:
                    if len(ALL_TICKERS) > 1:
                        df = raw_data[symbol].dropna()
                    else:
                        df = raw_data.dropna()

                    sig = get_signal(df, name)

                    if sig:
                        box_style = "box silver-box" if "SILVER" in name else "box"

                        with cols[idx % 3]:
                            st.markdown(f"""
                            <div class="{box_style}">
                                <div style="display:flex; justify-content:space-between;">
                                    <b>{name}</b>
                                    <span style="color:{sig['color']}; font-weight:bold;">{sig['action']}</span>
                                </div>
                                <h2 style="margin:5px 0;">{sig['price']:.2f}</h2>
                                <div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:5px;">
                                    <span>💪 Strength: <b>{sig['strength']}</b> ({sig['adx']:.0f})</span>
                                    <span>📊 Z-Score: <b>{sig['z_score']:.2f}</b></span>
                                </div>
                                <hr style="border-color:#333;">
                                <div style="font-size:14px;">
                                    🎯 Target: <span style="color:#00ff00;">{sig['tgt']:.2f}</span><br>
                                    🛑 StopLoss: <span style="color:#ff0000;">{sig['sl']:.2f}</span><br>
                                    🧱 Resistance: <span style="color:orange;">{sig['res']:.2f}</span><br>
                                    🛏️ Support: <span style="color:skyblue;">{sig['supp']:.2f}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        idx += 1
                except Exception:
                    continue
            st.markdown("---")

# === TAB 2: NEWS & SENTIMENT ===
with tab2:
    st.header("📰 Global News & AI Analysis")

    st.subheader("🌍 Global Fear Meter (Macro)")
    m_cols = st.columns(4)
    macro_ticks = ["^INDIAVIX", "^VIX", "DX-Y.NYB", "BZ=F"]
    macro_names = ["🇮🇳 INDIA VIX", "🇺🇸 US VIX", "💵 DOLLAR (DXY)", "🛢️ BRENT OIL"]

    try:
        m_data = yf.download(macro_ticks, period="5d", interval="1d", group_by='ticker', progress=False)
        for i, tick in enumerate(macro_ticks):
            try:
                curr = m_data[tick]['Close'].iloc[-1]
                prev = m_data[tick]['Close'].iloc[-2]
                chg = ((curr - prev) / prev) * 100
                colr = "red" if chg > 0 else "green"
                with m_cols[i]:
                    st.markdown(f"""
                    <div class="box" style="text-align:center;">
                        <small>{macro_names[i]}</small><br>
                        <b style="font-size:20px;">{curr:.2f}</b><br>
                        <span style="color:{colr};">{chg:+.2f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception:
                pass
    except Exception:
        st.error("Macro Data Load Fail")

    st.markdown("---")

    # FIX: Removed the stray HTML block (lines 273-334 in original) that was
    # accidentally pasted into the Python file here.

    news_opt = st.radio("Select News Source:", ["General", "Crypto", "India"], horizontal=True)

    feed = feedparser.parse(RSS_FEEDS[news_opt])
    for entry in feed.entries[:10]:
        sent, badge_cls = analyze_news_sentiment(entry.title)
        impact = get_impact(entry.title)

        st.markdown(f"""
        <div class="box">
            <div style="display:flex; justify-content:space-between;">
                <a href="{entry.link}" target="_blank" class="news-title">{entry.title}</a>
                <span class="{badge_cls}">{sent}</span>
            </div>
            <div style="color:#888; font-size:12px; margin-top:5px;">🕒 {entry.get('published', 'Just Now')}</div>
            <div class="impact-msg">{impact}</div>
        </div>
        """, unsafe_allow_html=True)
