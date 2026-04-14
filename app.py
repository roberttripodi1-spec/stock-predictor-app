from __future__ import annotations

import urllib.parse
import xml.etree.ElementTree as ET
from io import BytesIO

import pandas as pd
import plotly.graph_objects as go
import qrcode
import requests
import streamlit as st

from predictor import generate_projection_chart_data, screen_tickers, train_predict_for_ticker


APP_URL = "https://stock-predictor-app-chqgww4vn5xvfzytgesxvv.streamlit.app/"


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


@st.cache_data(ttl=3600, show_spinner=False)
def build_qr_code(url: str) -> bytes:
    qr = qrcode.make(url)
    buf = BytesIO()
    qr.save(buf, format="PNG")
    return buf.getvalue()


st.set_page_config(page_title="Stock Predictor", layout="wide")

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #0b1220 0%, #101828 100%);
        color: #f8fafc;
    }

    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }
    }

    h1, h2, h3 {
        color: #f8fafc !important;
        letter-spacing: -0.02em;
    }

    .top-card {
        padding: 1rem 1.1rem;
        border: 1px solid #223046;
        border-radius: 16px;
        background: #111827;
        box-shadow: 0 6px 20px rgba(0,0,0,0.18);
        margin-bottom: 1rem;
    }

    .signal-buy, .signal-sell, .signal-watch {
        padding: .85rem 1rem;
        border-radius: 12px;
        font-weight: 800;
        text-align: center;
        border: 1px solid transparent;
        margin-bottom: .75rem;
        letter-spacing: .02em;
    }

    .signal-buy { background: #0b3b2e; color: #6ee7b7; border-color: #14532d; }
    .signal-sell { background: #4c1717; color: #fca5a5; border-color: #7f1d1d; }
    .signal-watch { background: #5a3b10; color: #fcd34d; border-color: #92400e; }

    .small-note { color: #a5b4c7; font-size: .92rem; margin-top: .35rem; }
    .flag {
        padding: .42rem .66rem;
        border-radius: 999px;
        display: inline-block;
        margin: .18rem .22rem .18rem 0;
        background: #1b2638;
        border: 1px solid #314158;
        color: #e5edf8;
        font-size: .84rem;
    }
    .headline-card {
        padding: .8rem .9rem;
        border: 1px solid #223046;
        border-radius: 12px;
        background: #0f172a;
        margin-bottom: .6rem;
    }
    .headline-source {
        color: #93c5fd;
        font-size: .82rem;
        font-weight: 600;
    }

    [data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #223046;
        border-radius: 14px;
        padding: .65rem .8rem;
    }

    [data-testid="stMetricLabel"] { color: #b8c4d6 !important; font-weight: 600; }
    [data-testid="stMetricValue"] { color: #f8fafc !important; }

    div[data-testid="stDataFrame"] {
        border: 1px solid #223046;
        border-radius: 14px;
        overflow: hidden;
        background: #111827;
    }

    .stTabs [data-baseweb="tab-list"] { gap: .4rem; }
    .stTabs [data-baseweb="tab"] {
        background: #162032;
        border-radius: 10px 10px 0 0;
        color: #d9e3f0;
        padding: .45rem .85rem;
    }
    .stTabs [aria-selected="true"] { background: #24324a !important; }

    section[data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid #1f2a3d;
    }

    .stButton > button {
        background: #2563eb;
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 700;
    }
    .stButton > button:hover { background: #1d4ed8; color: white; }

    .stSelectbox label, .stTextInput label, .stSlider label {
        color: #dbe4f0 !important;
        font-weight: 600;
    }
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
        x=chart_data.index,
        open=chart_data["Open"],
        high=chart_data["High"],
        low=chart_data["Low"],
        close=chart_data["Close"],
        name="Price",
        increasing_line_color="#34d399",
        decreasing_line_color="#f87171",
        increasing_fillcolor="#34d399",
        decreasing_fillcolor="#f87171",
    ))
    fig.add_trace(go.Scatter(
        x=chart_data.index,
        y=chart_data["SMA20"],
        mode="lines",
        name="20D Avg",
        line=dict(color="#60a5fa", width=2),
    ))
    for y, label, color in [
        (current_price, "Current", "#cbd5e1"),
        (support, "Support", "#f59e0b"),
        (resistance, "Resistance", "#a78bfa"),
    ]:
        fig.add_hline(
            y=y,
            line_dash="dot",
            line_color=color,
            line_width=1.4,
            annotation_text=label,
            annotation_position="right",
            annotation_font_color=color,
        )

    fig.update_layout(
        title=f"{safe_attr(result, 'ticker', 'Ticker')} Price",
        height=500,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f172a",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=20, r=20, t=55, b=20),
        font=dict(color="#e5edf8"),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, zeroline=False, title=None)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)", zeroline=False, title=None)
    return fig


def build_projection_chart(summary: pd.DataFrame, current_price: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=summary.index,
        y=summary["High Band (90%)"],
        mode="lines",
        line=dict(color="rgba(96,165,250,0.0)", width=0),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=summary.index,
        y=summary["Low Band (10%)"],
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(96,165,250,0.18)",
        line=dict(color="rgba(96,165,250,0.0)", width=0),
        name="Projected Range",
    ))
    fig.add_trace(go.Scatter(
        x=summary.index,
        y=summary["Median"],
        mode="lines",
        name="Median Path",
        line=dict(color="#60a5fa", width=3),
    ))
    fig.add_hline(
        y=current_price,
        line_dash="dot",
        line_color="#cbd5e1",
        line_width=1.3,
        annotation_text="Current",
        annotation_position="right",
        annotation_font_color="#cbd5e1",
    )

    fig.update_layout(
        title="Projection Range",
        height=360,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f172a",
        margin=dict(l=20, r=20, t=55, b=20),
        font=dict(color="#e5edf8"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, title=None)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)", zeroline=False, title=None)
    return fig


def build_simple_gauge(title: str, value: float, min_value: float, max_value: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title},
        gauge={
            "axis": {"range": [min_value, max_value]},
            "bar": {"color": "#60a5fa"},
            "bgcolor": "#111827",
            "bordercolor": "#223046",
        }
    ))
    fig.update_layout(
        height=220,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=10),
        font=dict(color="#e5edf8"),
    )
    return fig


header_left, header_right = st.columns([3, 1])
with header_left:
    st.title("Stock Predictor")
    st.caption("A cleaner stock dashboard with ranked tickers, simpler charts, live headlines, projection ranges, and quick trading cues.")
with header_right:
    st.link_button("📲 Share App", APP_URL, use_container_width=True)

share_tab, app_tab = st.tabs(["Share", "Dashboard"])

with share_tab:
    st.subheader("Share with friends")
    st.write("Open this link on desktop or mobile, or scan the QR code below.")
    st.code(APP_URL, language=None)
    st.link_button("Open share link", APP_URL)
    qr_bytes = build_qr_code(APP_URL)
    st.image(qr_bytes, caption="Scan to open on your phone", width=220)

with app_tab:
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

            tabs = st.tabs(["Summary", "Latest headlines", "Feature importance"])

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
                feature_df = pd.DataFrame(safe_attr(result, "top_features", []), columns=["Feature", "Importance"])
                if not feature_df.empty:
                    feature_df["Importance"] = feature_df["Importance"].map(lambda x: f"{x:.3f}")
                    st.dataframe(feature_df, use_container_width=True, hide_index=True)
                else:
                    st.write("No feature importance data available.")

        st.caption("This version adds a top share button, a QR code share tab, and mobile-friendly spacing.")
    else:
        st.info("Enter your tickers on the left and click Run.")
