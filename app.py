from __future__ import annotations

import math
import urllib.parse
import xml.etree.ElementTree as ET

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

from predictor import generate_projection_chart_data, screen_tickers, train_predict_for_ticker


def safe_attr(obj, name, default):
    return getattr(obj, name, default)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_live_headlines(ticker: str, limit: int = 8) -> list[dict]:
    query = urllib.parse.quote(f"{ticker} stock")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

    items = []
    try:
        response = requests.get(url, timeout=12)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        channel = root.find("channel")
        if channel is None:
            return items

        for item in channel.findall("item")[:limit]:
            title = item.findtext("title", default="").strip()
            link = item.findtext("link", default="").strip()
            pub_date = item.findtext("pubDate", default="").strip()
            source_el = item.find("source")
            source = source_el.text.strip() if source_el is not None and source_el.text else ""
            if title:
                items.append({
                    "title": title,
                    "link": link,
                    "source": source,
                    "published": pub_date,
                })
    except Exception:
        return []

    return items


@st.cache_data(ttl=900, show_spinner=False)
def fetch_option_ideas(ticker: str, spot_price: float, signal: str, up_probability: float) -> dict:
    """
    Pull a near-term options chain and surface simple research ideas.
    This is intentionally simple and educational, not trade advice.
    """
    result = {
        "expiry": None,
        "bullish": [],
        "bearish": [],
        "notes": [],
    }
    try:
        tk = yf.Ticker(ticker)
        expirations = tk.options
        if not expirations:
            result["notes"].append("No option expirations found for this ticker.")
            return result

        expiry = expirations[0]
        chain = tk.option_chain(expiry)
        calls = chain.calls.copy()
        puts = chain.puts.copy()
        result["expiry"] = expiry

        # Clean
        for df in [calls, puts]:
            for col in ["strike", "lastPrice", "bid", "ask", "impliedVolatility", "volume", "openInterest"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

        # Mid price fallback
        def calc_mid(df):
            bid = df.get("bid", pd.Series([None]*len(df)))
            ask = df.get("ask", pd.Series([None]*len(df)))
            last = df.get("lastPrice", pd.Series([None]*len(df)))
            mid = ((bid.fillna(0) + ask.fillna(0)) / 2).replace(0, pd.NA)
            return mid.fillna(last)

        calls["premium"] = calc_mid(calls)
        puts["premium"] = calc_mid(puts)

        calls = calls.dropna(subset=["strike", "premium"]).copy()
        puts = puts.dropna(subset=["strike", "premium"]).copy()

        if calls.empty or puts.empty:
            result["notes"].append("Option chain loaded, but pricing data was incomplete.")
            return result

        # Nearby ATM and slightly OTM ideas
        calls["distance"] = (calls["strike"] - spot_price).abs()
        puts["distance"] = (puts["strike"] - spot_price).abs()

        atm_call = calls.sort_values("distance").head(1)
        atm_put = puts.sort_values("distance").head(1)

        otm_call = calls[calls["strike"] >= spot_price * 1.02].sort_values("strike").head(1)
        otm_put = puts[puts["strike"] <= spot_price * 0.98].sort_values("strike", ascending=False).head(1)

        def make_contract_idea(row, side: str, style: str):
            strike = float(row["strike"])
            premium = float(row["premium"]) if pd.notnull(row["premium"]) else 0.0
            iv = float(row["impliedVolatility"]) if pd.notnull(row.get("impliedVolatility", None)) else float("nan")
            oi = int(row["openInterest"]) if pd.notnull(row.get("openInterest", None)) else 0
            vol = int(row["volume"]) if pd.notnull(row.get("volume", None)) else 0

            if side == "call":
                breakeven = strike + premium
                thesis = "Bullish exposure"
            else:
                breakeven = strike - premium
                thesis = "Bearish exposure"

            label = f"{style} {side.upper()} {int(strike) if strike.is_integer() else strike}"
            return {
                "label": label,
                "strike": strike,
                "premium": premium,
                "breakeven": breakeven,
                "max_risk": premium * 100,
                "iv": iv,
                "open_interest": oi,
                "volume": vol,
                "thesis": thesis,
            }

        bullish = []
        bearish = []

        if not atm_call.empty:
            bullish.append(make_contract_idea(atm_call.iloc[0], "call", "ATM"))
        if not otm_call.empty:
            bullish.append(make_contract_idea(otm_call.iloc[0], "call", "OTM"))
        if not atm_put.empty:
            bearish.append(make_contract_idea(atm_put.iloc[0], "put", "ATM"))
        if not otm_put.empty:
            bearish.append(make_contract_idea(otm_put.iloc[0], "put", "OTM"))

        # Add short text note based on model tilt
        if signal == "BUY" or up_probability >= 0.58:
            result["notes"].append("Model leans bullish, so calls are more aligned with the current signal than puts.")
        elif signal == "SELL" or up_probability <= 0.42:
            result["notes"].append("Model leans bearish, so puts are more aligned with the current signal than calls.")
        else:
            result["notes"].append("Model signal is mixed, so both calls and puts should be treated as watchlist ideas, not strong setups.")

        result["notes"].append("Max risk for a long call or long put is the premium paid.")
        result["bullish"] = bullish
        result["bearish"] = bearish
        return result
    except Exception as e:
        result["notes"].append(f"Could not load options chain right now.")
        return result


st.set_page_config(page_title="Stock Predictor", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg, #0b1220 0%, #101828 100%); color: #f8fafc; }
    .main .block-container { padding-top: 1.6rem; padding-bottom: 2rem; }
    h1, h2, h3 { color: #f8fafc !important; letter-spacing: -0.02em; }
    .top-card { padding: 1rem 1.1rem; border: 1px solid #223046; border-radius: 16px; background: #111827; box-shadow: 0 6px 20px rgba(0,0,0,0.18); margin-bottom: 1rem; }
    .signal-buy, .signal-sell, .signal-watch { padding: .85rem 1rem; border-radius: 12px; font-weight: 800; text-align: center; border: 1px solid transparent; margin-bottom: .75rem; letter-spacing: .02em; }
    .signal-buy { background: #0b3b2e; color: #6ee7b7; border-color: #14532d; }
    .signal-sell { background: #4c1717; color: #fca5a5; border-color: #7f1d1d; }
    .signal-watch { background: #5a3b10; color: #fcd34d; border-color: #92400e; }
    .small-note { color: #a5b4c7; font-size: .92rem; margin-top: .35rem; }
    .flag { padding: .42rem .66rem; border-radius: 999px; display: inline-block; margin: .18rem .22rem .18rem 0; background: #1b2638; border: 1px solid #314158; color: #e5edf8; font-size: .84rem; }
    .headline-card, .option-card {
        padding: .85rem .95rem; border: 1px solid #223046; border-radius: 12px; background: #0f172a; margin-bottom: .65rem;
    }
    .headline-source, .option-label { color: #93c5fd; font-size: .82rem; font-weight: 700; }
    .option-grid { color:#dbe4f0; font-size:.9rem; line-height:1.55; margin-top:.35rem; }
    [data-testid="stMetric"] { background: #111827; border: 1px solid #223046; border-radius: 14px; padding: .65rem .8rem; }
    [data-testid="stMetricLabel"] { color: #b8c4d6 !important; font-weight: 600; }
    [data-testid="stMetricValue"] { color: #f8fafc !important; }
    div[data-testid="stDataFrame"] { border: 1px solid #223046; border-radius: 14px; overflow: hidden; background: #111827; }
    .stTabs [data-baseweb="tab-list"] { gap: .4rem; }
    .stTabs [data-baseweb="tab"] { background: #162032; border-radius: 10px 10px 0 0; color: #d9e3f0; padding: .45rem .85rem; }
    .stTabs [aria-selected="true"] { background: #24324a !important; }
    section[data-testid="stSidebar"] { background: #0f172a; border-right: 1px solid #1f2a3d; }
    .stButton > button { background: #2563eb; color: white; border: none; border-radius: 10px; font-weight: 700; }
    .stButton > button:hover { background: #1d4ed8; color: white; }
    .stSelectbox label, .stTextInput label, .stSlider label { color: #dbe4f0 !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


def signal_class(signal: str) -> str:
    return {"BUY": "signal-buy", "SELL": "signal-sell"}.get(signal, "signal-watch")


def build_candlestick_chart(hist: pd.DataFrame, result) -> go.Figure:
    chart_data = hist.tail(90).copy()
    chart_data["SMA20"] = chart_data["Close"].rolling(20).mean()

    current_price = safe_attr(result, "latest_close", float(chart_data["Close"].iloc[-1]))
    support = safe_attr(result, "support_level", float(chart_data["Low"].tail(30).min()))
    resistance = safe_attr(result, "resistance_level", float(chart_data["High"].tail(30).max()))

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=chart_data.index, open=chart_data["Open"], high=chart_data["High"], low=chart_data["Low"], close=chart_data["Close"],
        name="Price", increasing_line_color="#34d399", decreasing_line_color="#f87171",
        increasing_fillcolor="#34d399", decreasing_fillcolor="#f87171",
    ))
    fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data["SMA20"], mode="lines", name="20D Avg", line=dict(color="#60a5fa", width=2)))
    for y, label, color in [(current_price, "Current", "#cbd5e1"), (support, "Support", "#f59e0b"), (resistance, "Resistance", "#a78bfa")]:
        fig.add_hline(y=y, line_dash="dot", line_color=color, line_width=1.4, annotation_text=label, annotation_position="right", annotation_font_color=color)

    fig.update_layout(
        title=f"{safe_attr(result, 'ticker', 'Ticker')} Price", height=500, xaxis_rangeslider_visible=False, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f172a",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=20, r=20, t=55, b=20), font=dict(color="#e5edf8"), hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, zeroline=False, title=None)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)", zeroline=False, title=None)
    return fig


def build_projection_chart(summary: pd.DataFrame, current_price: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=summary.index, y=summary["High Band (90%)"], mode="lines", line=dict(color="rgba(96,165,250,0.0)", width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=summary.index, y=summary["Low Band (10%)"], mode="lines", fill="tonexty", fillcolor="rgba(96,165,250,0.18)", line=dict(color="rgba(96,165,250,0.0)", width=0), name="Projected Range"))
    fig.add_trace(go.Scatter(x=summary.index, y=summary["Median"], mode="lines", name="Median Path", line=dict(color="#60a5fa", width=3)))
    fig.add_hline(y=current_price, line_dash="dot", line_color="#cbd5e1", line_width=1.3, annotation_text="Current", annotation_position="right", annotation_font_color="#cbd5e1")
    fig.update_layout(
        title="Projection Range", height=360, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f172a",
        margin=dict(l=20, r=20, t=55, b=20), font=dict(color="#e5edf8"), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, title=None)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)", zeroline=False, title=None)
    return fig


def build_simple_gauge(title: str, value: float, min_value: float, max_value: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, title={"text": title},
        gauge={"axis": {"range": [min_value, max_value]}, "bar": {"color": "#60a5fa"}, "bgcolor": "#111827", "bordercolor": "#223046"}
    ))
    fig.update_layout(height=220, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=40, b=10), font=dict(color="#e5edf8"))
    return fig


st.title("Stock Predictor")
st.caption("A cleaner stock dashboard with ranked tickers, simpler charts, live headlines, projection ranges, and research-based options ideas.")

with st.sidebar:
    st.header("Inputs")
    tickers_text = st.text_input("Ticker list", value="AAPL, MSFT, NVDA, SPY, TSLA")
    selected_ticker = st.text_input("Main ticker to view", value="AAPL")
    period = st.selectbox("History period", options=["1y", "2y", "5y", "10y"], index=2)
    threshold = st.slider("Signal threshold", min_value=0.50, max_value=0.75, value=0.55, step=0.01)
    forecast_days = st.slider("Projection days", min_value=5, max_value=60, value=20, step=5)
    n_sims = st.slider("Projection paths", min_value=50, max_value=500, value=200, step=50)
    run = st.button("Run", use_container_width=True)

if run:
    tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
    with st.spinner("Loading market view..."):
        df = screen_tickers(tickers, period=period, threshold=threshold)

    st.markdown('<div class="top-card">', unsafe_allow_html=True)
    st.subheader("Watchlist")
    display_df = df.copy()
    for col in ["Latest Close"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")
    for col in ["Up Probability", "Holdout Accuracy", "20D Momentum", "Strategy Return", "Buy & Hold Return"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
    for col in ["RSI", "Volume Ratio"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda x: f"{x:,.2f}" if pd.notnull(x) else "")
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.markdown('<div class="small-note">Choose the ticker you want to study in the sidebar under "Main ticker to view".</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    focus = selected_ticker.strip().upper() if selected_ticker.strip() else None
    if focus not in tickers and len(tickers) > 0:
        focus = tickers[0]

    result = train_predict_for_ticker(focus, period=period, threshold=threshold)
    summary, _ = generate_projection_chart_data(result, forecast_days=forecast_days, n_sims=n_sims)
    live_headlines = fetch_live_headlines(focus, limit=8)
    option_ideas = fetch_option_ideas(
        focus,
        spot_price=float(safe_attr(result, "latest_close", 0.0)),
        signal=str(safe_attr(result, "model_signal", "WATCH")),
        up_probability=float(safe_attr(result, "next_day_up_probability", 0.5)),
    )

    left, right = st.columns([1, 2])

    with left:
        st.markdown(f'<div class="{signal_class(safe_attr(result, "model_signal", "WATCH"))}">{safe_attr(result, "model_signal", "WATCH")}</div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        a.metric("Price", f"${safe_attr(result, 'latest_close', 0.0):,.2f}")
        b.metric("Mood", safe_attr(result, "mood", "Neutral"))
        c, d = st.columns(2)
        c.metric("Up probability", f"{safe_attr(result, 'next_day_up_probability', 0.5):.2%}")
        d.metric("Accuracy", f"{safe_attr(result, 'holdout_accuracy', 0.0):.2%}")
        e, f = st.columns(2)
        e.metric("Momentum", f"{safe_attr(result, 'momentum_20d', 0.0):.2%}")
        f.metric("Volume", f"{safe_attr(result, 'volume_ratio', 1.0):.2f}x")
        g, h = st.columns(2)
        g.metric("News tone", safe_attr(result, "sentiment_label", "Neutral"))
        h.metric("Earnings", safe_attr(result, "earnings_flag", "No date found"))

        st.plotly_chart(build_simple_gauge("RSI", max(0, min(100, safe_attr(result, "rsi_14", 50.0))), 0, 100), use_container_width=True)
        st.plotly_chart(build_simple_gauge("News Sentiment", max(-1, min(1, safe_attr(result, "sentiment_score", 0.0))), -1, 1), use_container_width=True)

        st.subheader("Key levels")
        levels_df = pd.DataFrame({
            "Item": ["Support", "Resistance", "Stop loss", "Target 1", "Target 2"],
            "Value": [
                safe_attr(result, "support_level", 0.0),
                safe_attr(result, "resistance_level", 0.0),
                safe_attr(result, "stop_loss", 0.0),
                safe_attr(result, "target_1", 0.0),
                safe_attr(result, "target_2", 0.0),
            ],
        })
        levels_df["Value"] = levels_df["Value"].map(lambda x: f"${x:,.2f}")
        st.dataframe(levels_df, use_container_width=True, hide_index=True)

        st.subheader("Flags")
        flags = safe_attr(result, "watchlist_flags", ["No major alert flags"])
        flags_html = "".join([f'<span class="flag">{flag}</span>' for flag in flags])
        st.markdown(flags_html, unsafe_allow_html=True)

    with right:
        st.plotly_chart(build_candlestick_chart(result.history, result), use_container_width=True)
        st.plotly_chart(build_projection_chart(summary, safe_attr(result, "latest_close", 0.0)), use_container_width=True)

        tabs = st.tabs(["Summary", "Latest headlines", "Suggested options plays", "Feature importance"])

        with tabs[0]:
            notes = []
            rsi = safe_attr(result, "rsi_14", 50.0)
            notes.append(f"Signal: {safe_attr(result, 'model_signal', 'WATCH')}")
            notes.append(f"RSI is {'high' if rsi > 70 else 'low' if rsi < 40 else 'middle-range'} at {rsi:.1f}.")
            notes.append(f"Price is {'above' if safe_attr(result, 'latest_close', 0.0) > safe_attr(result, 'sma20', 0.0) else 'below'} the 20-day average.")
            notes.append(f"MACD is {'above' if safe_attr(result, 'macd', 0.0) > safe_attr(result, 'macd_signal', 0.0) else 'below'} its signal line.")
            notes.append(f"Earnings status: {safe_attr(result, 'earnings_flag', 'No date found')}.")
            for note in notes:
                st.write(f"- {note}")

        with tabs[1]:
            if live_headlines:
                st.caption(f"Latest headlines for {focus}. Refresh or rerun later for newer items.")
                for item in live_headlines:
                    source = item.get("source", "")
                    published = item.get("published", "")
                    title = item.get("title", "")
                    link = item.get("link", "")
                    source_line = f"{source} • {published}" if source and published else source or published
                    st.markdown(
                        f"""
                        <div class="headline-card">
                            <div class="headline-source">{source_line}</div>
                            <div style="margin-top:.28rem;">
                                <a href="{link}" target="_blank" style="color:#f8fafc; text-decoration:none; font-weight:600;">{title}</a>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.write("No live headlines were returned right now. Try rerunning in a few minutes.")

        with tabs[2]:
            expiry = option_ideas.get("expiry")
            if expiry:
                st.caption(f"Option ideas based on the nearest listed expiration: {expiry}. These are research ideas, not trade advice.")
            else:
                st.caption("Options ideas are research-only and depend on live option chain availability.")

            notes = option_ideas.get("notes", [])
            for note in notes:
                st.write(f"- {note}")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Call ideas")
                bullish = option_ideas.get("bullish", [])
                if bullish:
                    for idea in bullish:
                        st.markdown(
                            f"""
                            <div class="option-card">
                                <div class="option-label">{idea['label']}</div>
                                <div class="option-grid">
                                    Thesis: {idea['thesis']}<br>
                                    Strike: ${idea['strike']:,.2f}<br>
                                    Est. premium: ${idea['premium']:,.2f}<br>
                                    Breakeven: ${idea['breakeven']:,.2f}<br>
                                    Max risk: ${idea['max_risk']:,.0f}<br>
                                    Open interest: {idea['open_interest']:,}<br>
                                    Volume: {idea['volume']:,}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.write("No call ideas were returned.")

            with col2:
                st.subheader("Put ideas")
                bearish = option_ideas.get("bearish", [])
                if bearish:
                    for idea in bearish:
                        st.markdown(
                            f"""
                            <div class="option-card">
                                <div class="option-label">{idea['label']}</div>
                                <div class="option-grid">
                                    Thesis: {idea['thesis']}<br>
                                    Strike: ${idea['strike']:,.2f}<br>
                                    Est. premium: ${idea['premium']:,.2f}<br>
                                    Breakeven: ${idea['breakeven']:,.2f}<br>
                                    Max risk: ${idea['max_risk']:,.0f}<br>
                                    Open interest: {idea['open_interest']:,}<br>
                                    Volume: {idea['volume']:,}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.write("No put ideas were returned.")

        with tabs[3]:
            feature_df = pd.DataFrame(safe_attr(result, "top_features", []), columns=["Feature", "Importance"])
            if not feature_df.empty:
                feature_df["Importance"] = feature_df["Importance"].map(lambda x: f"{x:.3f}")
                st.dataframe(feature_df, use_container_width=True, hide_index=True)
            else:
                st.write("No feature importance data available.")

    st.caption("Suggested options plays are simple long call/put research ideas from the live options chain for the chosen ticker.")
else:
    st.info("Enter your tickers on the left and click Run.")
