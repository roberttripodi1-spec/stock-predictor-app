from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from predictor import generate_projection_chart_data, screen_tickers, train_predict_for_ticker


st.set_page_config(page_title="Stock Predictor Elite", layout="wide")

st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at top left, #0f172a 0%, #07111f 40%, #020617 100%);
        color: #e5e7eb;
    }
    .hero {
        padding: 1.25rem 1.4rem;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(37,99,235,0.20), rgba(168,85,247,0.18), rgba(16,185,129,0.14));
        box-shadow: 0 12px 40px rgba(0,0,0,0.28);
        margin-bottom: 1rem;
    }
    .pill-row { display:flex; gap:.5rem; flex-wrap:wrap; margin-top:.8rem; }
    .pill {
        padding:.35rem .7rem; border-radius:999px; background:rgba(255,255,255,0.08);
        color:#e2e8f0; font-size:.88rem; border:1px solid rgba(255,255,255,0.07);
    }
    .signal-buy, .signal-sell, .signal-watch {
        padding: .9rem 1rem; border-radius: 14px; font-weight: 800; text-align:center;
        border:1px solid rgba(255,255,255,0.08); margin-bottom: .75rem;
    }
    .signal-buy { background: rgba(16,185,129,0.15); color: #a7f3d0; }
    .signal-sell { background: rgba(239,68,68,0.16); color: #fecaca; }
    .signal-watch { background: rgba(245,158,11,0.16); color: #fde68a; }
    .flag {
        padding: .45rem .65rem; border-radius: 999px; display: inline-block; margin: .2rem .25rem .2rem 0;
        background: rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.07); font-size: .88rem;
    }
</style>
""", unsafe_allow_html=True)

def build_candlestick_chart(hist: pd.DataFrame, result) -> go.Figure:
    chart_data = hist.tail(120).copy()
    chart_data["SMA20"] = chart_data["Close"].rolling(20).mean()
    chart_data["SMA50"] = chart_data["Close"].rolling(50).mean()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=chart_data.index,
        open=chart_data["Open"],
        high=chart_data["High"],
        low=chart_data["Low"],
        close=chart_data["Close"],
        name="Price",
    ))
    fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data["SMA20"], mode="lines", name="SMA 20"))
    fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data["SMA50"], mode="lines", name="SMA 50"))
    fig.add_hline(y=result.support_level, line_dash="dash", annotation_text="Support")
    fig.add_hline(y=result.resistance_level, line_dash="dash", annotation_text="Resistance")
    fig.add_hline(y=result.stop_loss, line_dash="dot", annotation_text="Stop")
    fig.add_hline(y=result.target_1, line_dash="dot", annotation_text="T1")
    fig.add_hline(y=result.target_2, line_dash="dot", annotation_text="T2")
    fig.update_layout(
        title=f"{result.ticker} Price Structure",
        height=620,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def build_projection_chart(summary: pd.DataFrame, current_price: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=summary.index, y=summary["High Band (90%)"], mode="lines", name="Upper Band"))
    fig.add_trace(go.Scatter(x=summary.index, y=summary["Low Band (10%)"], mode="lines", name="Lower Band", fill="tonexty"))
    fig.add_trace(go.Scatter(x=summary.index, y=summary["Median"], mode="lines", name="Median"))
    fig.add_hline(y=current_price, line_dash="dash", annotation_text="Current Price")
    fig.update_layout(
        title="Projected Price Cone",
        height=480,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def build_rsi_gauge(result) -> go.Figure:
    rsi_value = max(0, min(100, result.rsi_14))
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rsi_value,
        title={"text": "RSI Momentum"},
        gauge={
            "axis": {"range": [0, 100]},
            "steps": [
                {"range": [0, 30], "color": "rgba(239,68,68,0.35)"},
                {"range": [30, 70], "color": "rgba(59,130,246,0.25)"},
                {"range": [70, 100], "color": "rgba(245,158,11,0.35)"},
            ],
            "threshold": {"line": {"color": "white", "width": 4}, "thickness": 0.75, "value": rsi_value},
        }
    ))
    fig.update_layout(height=260, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=20,r=20,t=40,b=10))
    return fig


def build_sentiment_gauge(result) -> go.Figure:
    value = max(-1, min(1, result.sentiment_score))
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": ""},
        title={"text": "News Sentiment"},
        gauge={
            "axis": {"range": [-1, 1]},
            "steps": [
                {"range": [-1, -0.25], "color": "rgba(239,68,68,0.35)"},
                {"range": [-0.25, 0.25], "color": "rgba(148,163,184,0.25)"},
                {"range": [0.25, 1], "color": "rgba(16,185,129,0.30)"},
            ],
            "threshold": {"line": {"color": "white", "width": 4}, "thickness": 0.75, "value": value},
        }
    ))
    fig.update_layout(height=260, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=20,r=20,t=40,b=10))
    return fig


def signal_class(signal: str) -> str:
    return {"BUY": "signal-buy", "SELL": "signal-sell"}.get(signal, "signal-watch")


st.markdown("""
<div class="hero">
    <h1 style="margin:0;">Stock Predictor Elite</h1>
    <p style="margin:.35rem 0 0 0; color:#cbd5e1;">Styled research terminal with price structure, projection cones, earnings awareness, and headline tone.</p>
    <div class="pill-row">
        <div class="pill">Signal engine</div>
        <div class="pill">Projection cone</div>
        <div class="pill">News sentiment</div>
        <div class="pill">Earnings risk</div>
        <div class="pill">Watchlist alerts</div>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Control Panel")
    tickers_text = st.text_input("Ticker List", value="AAPL, MSFT, NVDA, SPY, TSLA")
    period = st.selectbox("History Period", options=["1y", "2y", "5y", "10y"], index=2)
    threshold = st.slider("Signal Threshold", min_value=0.50, max_value=0.75, value=0.55, step=0.01)
    forecast_days = st.slider("Projection Days", min_value=5, max_value=60, value=20, step=5)
    n_sims = st.slider("Simulation Paths", min_value=50, max_value=500, value=200, step=50)
    selected_ticker = st.text_input("Focus Ticker", value="AAPL")
    run = st.button("Run Dashboard", use_container_width=True)

if run:
    tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
    with st.spinner("Building elite market view..."):
        df = screen_tickers(tickers, period=period, threshold=threshold)

    st.subheader("Ranked Watchlist")
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
    st.dataframe(display_df, use_container_width=True)

    focus = selected_ticker.strip().upper() if selected_ticker.strip() else None
    if focus not in tickers and len(tickers) > 0:
        focus = tickers[0]

    result = train_predict_for_ticker(focus, period=period, threshold=threshold)
    summary, _ = generate_projection_chart_data(result, forecast_days=forecast_days, n_sims=n_sims)

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.markdown(f'<div class="{signal_class(result.model_signal)}">{result.model_signal}</div>', unsafe_allow_html=True)
    col_b.metric("Market Mood", result.mood)
    col_c.metric("News Tone", result.sentiment_label)
    col_d.metric("Earnings Status", result.earnings_flag)

    row1 = st.columns(5)
    row1[0].metric("Latest Close", f"${result.latest_close:,.2f}")
    row1[1].metric("Up Probability", f"{result.next_day_up_probability:.2%}")
    row1[2].metric("20D Momentum", f"{result.momentum_20d:.2%}")
    row1[3].metric("Volume Ratio", f"{result.volume_ratio:.2f}x")
    row1[4].metric("52W Range Pos.", f"{result.range_52w_position:.0%}")

    row2 = st.columns(5)
    row2[0].metric("Holdout Accuracy", f"{result.holdout_accuracy:.2%}")
    row2[1].metric("Support", f"${result.support_level:,.2f}")
    row2[2].metric("Resistance", f"${result.resistance_level:,.2f}")
    row2[3].metric("Target 1", f"${result.target_1:,.2f}")
    row2[4].metric("Target 2", f"${result.target_2:,.2f}")

    left, right = st.columns([1.05, 2.1])

    with left:
        st.plotly_chart(build_rsi_gauge(result), use_container_width=True)
        st.plotly_chart(build_sentiment_gauge(result), use_container_width=True)

        st.subheader("Watchlist Flags")
        flags_html = "".join([f'<span class="flag">{flag}</span>' for flag in result.watchlist_flags])
        st.markdown(flags_html, unsafe_allow_html=True)

        st.subheader("Earnings Window")
        if result.earnings_date:
            st.write(f"Next/Recent earnings date: **{result.earnings_date}**")
        else:
            st.write("No earnings date found from the data source.")
        if result.days_to_earnings is not None:
            st.write(f"Days to earnings: **{result.days_to_earnings}**")

        st.subheader("Headline Tone")
        if result.headlines:
            for headline in result.headlines[:6]:
                st.write(f"- {headline}")
        else:
            st.write("No headlines returned by the data source.")

    with right:
        st.plotly_chart(build_candlestick_chart(result.history, result), use_container_width=True)
        st.plotly_chart(build_projection_chart(summary, result.latest_close), use_container_width=True)

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        st.subheader("Signal Notes")
        notes = []
        notes.append(f"RSI is {'overheated' if result.rsi_14 > 70 else 'weak' if result.rsi_14 < 40 else 'balanced'} at {result.rsi_14:.1f}.")
        notes.append(f"Price is {'above' if result.latest_close > result.sma20 else 'below'} the 20-day average and {'above' if result.latest_close > result.sma50 else 'below'} the 50-day average.")
        notes.append(f"MACD is {'above' if result.macd > result.macd_signal else 'below'} its signal line.")
        notes.append(f"Earnings status is {result.earnings_flag.lower()}.")
        notes.append(f"Headline tone is {result.sentiment_label.lower()} based on {result.headline_count} recent headlines.")
        for n in notes:
            st.write(f"- {n}")

    with bottom_right:
        st.subheader("Feature Importance")
        feature_df = pd.DataFrame(result.top_features, columns=["Feature", "Importance"])
        feature_df["Importance"] = feature_df["Importance"].map(lambda x: f"{x:.3f}")
        st.dataframe(feature_df, use_container_width=True, hide_index=True)

    st.caption(
        "This dashboard uses yfinance market and news feeds when available. Headline sentiment is lightweight keyword scoring, and earnings dates depend on the upstream data feed."
    )
