from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from predictor import generate_projection_chart_data, screen_tickers, train_predict_for_ticker


st.set_page_config(page_title="Stock Predictor Pro+", layout="wide")

st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at top left, #111827 0%, #0b1220 45%, #050816 100%);
        color: #e5e7eb;
    }
    .hero {
        padding: 1.2rem 1.4rem;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(37,99,235,0.18), rgba(16,185,129,0.12));
        box-shadow: 0 12px 40px rgba(0,0,0,0.25);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2rem;
    }
    .hero p {
        margin: 0.35rem 0 0 0;
        color: #cbd5e1;
    }
    .pill-row {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 0.8rem;
    }
    .pill {
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        color: #e2e8f0;
        font-size: 0.88rem;
        border: 1px solid rgba(255,255,255,0.07);
    }
    .section-card {
        padding: 1rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.03);
        margin-bottom: 1rem;
    }
    .signal-buy, .signal-sell, .signal-watch {
        padding: 0.8rem 1rem;
        border-radius: 14px;
        font-weight: 700;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .signal-buy { background: rgba(16,185,129,0.15); color: #a7f3d0; }
    .signal-sell { background: rgba(239,68,68,0.16); color: #fecaca; }
    .signal-watch { background: rgba(245,158,11,0.16); color: #fde68a; }
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
        height=500,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def build_momentum_gauge(result) -> go.Figure:
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
    fig.update_layout(
        height=280,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=10),
    )
    return fig


def signal_class(signal: str) -> str:
    return {
        "BUY": "signal-buy",
        "SELL": "signal-sell",
    }.get(signal, "signal-watch")


st.markdown("""
<div class="hero">
    <h1>Stock Predictor Pro+</h1>
    <p>Research dashboard with projection cones, signal cards, momentum context, and cleaner trading visuals.</p>
    <div class="pill-row">
        <div class="pill">Next-day probability model</div>
        <div class="pill">Projection scenarios</div>
        <div class="pill">Trend and momentum context</div>
        <div class="pill">Support / resistance / targets</div>
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
    with st.spinner("Building market view..."):
        df = screen_tickers(tickers, period=period, threshold=threshold)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
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
    st.markdown('</div>', unsafe_allow_html=True)

    focus = selected_ticker.strip().upper() if selected_ticker.strip() else None
    if focus not in tickers and len(tickers) > 0:
        focus = tickers[0]

    result = train_predict_for_ticker(focus, period=period, threshold=threshold)
    summary, _ = generate_projection_chart_data(result, forecast_days=forecast_days, n_sims=n_sims)

    left, right = st.columns([1.2, 2.2])

    with left:
        st.markdown(f'<div class="{signal_class(result.model_signal)}">{result.model_signal}</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Market Mood", result.mood)
        c2.metric("Up Probability", f"{result.next_day_up_probability:.2%}")

        c3, c4 = st.columns(2)
        c3.metric("Latest Close", f"${result.latest_close:,.2f}")
        c4.metric("Holdout Accuracy", f"{result.holdout_accuracy:.2%}")

        c5, c6 = st.columns(2)
        c5.metric("20D Momentum", f"{result.momentum_20d:.2%}")
        c6.metric("20D Volatility", f"{result.volatility_20:.2%}")

        c7, c8 = st.columns(2)
        c7.metric("Volume Ratio", f"{result.volume_ratio:.2f}x")
        c8.metric("52W Range Pos.", f"{result.range_52w_position:.0%}")

        c9, c10 = st.columns(2)
        c9.metric("Risk/Reward T1", f"{result.rr_1:.2f}")
        c10.metric("Risk/Reward T2", f"{result.rr_2:.2f}")

        st.plotly_chart(build_momentum_gauge(result), use_container_width=True)

        levels_df = pd.DataFrame({
            "Level": ["Support", "Resistance", "Stop Loss", "Target 1", "Target 2", "SMA 20", "SMA 50"],
            "Price": [
                result.support_level,
                result.resistance_level,
                result.stop_loss,
                result.target_1,
                result.target_2,
                result.sma20,
                result.sma50,
            ],
        })
        levels_df["Price"] = levels_df["Price"].map(lambda x: f"${x:,.2f}")
        st.subheader("Key Levels")
        st.dataframe(levels_df, use_container_width=True, hide_index=True)

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
        notes.append(f"20-day momentum sits at {result.momentum_20d:.2%}.")
        notes.append(f"Volume is running at {result.volume_ratio:.2f}x the 20-day average.")
        for n in notes:
            st.write(f"- {n}")

    with bottom_right:
        st.subheader("Feature Importance")
        feature_df = pd.DataFrame(result.top_features, columns=["Feature", "Importance"])
        feature_df["Importance"] = feature_df["Importance"].map(lambda x: f"{x:.3f}")
        st.dataframe(feature_df, use_container_width=True, hide_index=True)

    st.caption(
        "This is a model-driven research dashboard. It adds style and context, but it still does not guarantee future prices or profitable trades."
    )
