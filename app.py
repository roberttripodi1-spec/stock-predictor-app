from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from predictor import generate_projection_chart_data, screen_tickers, train_predict_for_ticker


def build_candlestick_chart(hist: pd.DataFrame, result) -> go.Figure:
    chart_data = hist.tail(90).copy()

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=chart_data.index,
            open=chart_data["Open"],
            high=chart_data["High"],
            low=chart_data["Low"],
            close=chart_data["Close"],
            name="Price",
        )
    )

    fig.add_hline(y=result.support_level, line_dash="dash", annotation_text="Support")
    fig.add_hline(y=result.resistance_level, line_dash="dash", annotation_text="Resistance")
    fig.add_hline(y=result.stop_loss, line_dash="dot", annotation_text="Stop Loss")
    fig.add_hline(y=result.target_1, line_dash="dot", annotation_text="Target 1")
    fig.add_hline(y=result.target_2, line_dash="dot", annotation_text="Target 2")

    fig.update_layout(
        title=f"{result.ticker} Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        height=600,
        xaxis_rangeslider_visible=False,
    )
    return fig


def build_projection_chart(summary: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=summary.index, y=summary["Median"], mode="lines", name="Median Projection"))
    fig.add_trace(go.Scatter(x=summary.index, y=summary["High Band (90%)"], mode="lines", name="High Band (90%)"))
    fig.add_trace(go.Scatter(x=summary.index, y=summary["Low Band (10%)"], mode="lines", name="Low Band (10%)", fill="tonexty"))
    fig.add_trace(go.Scatter(x=summary.index, y=summary["Bull Case (95%)"], mode="lines", name="Bull Case (95%)"))
    fig.add_trace(go.Scatter(x=summary.index, y=summary["Bear Case (5%)"], mode="lines", name="Bear Case (5%)"))

    fig.update_layout(
        title="Projection Chart",
        xaxis_title="Date",
        yaxis_title="Projected Price",
        height=500,
    )
    return fig


def signal_box(signal: str) -> str:
    if signal == "BUY":
        return "🟢 BUY"
    if signal == "SELL":
        return "🔴 SELL"
    return "🟡 WATCH"


st.set_page_config(page_title="Stock Predictor Pro", layout="wide")
st.title("Stock Predictor Pro")
st.caption("Trading-style dashboard with ranking, candlesticks, price levels, and simulated projections.")

with st.sidebar:
    tickers_text = st.text_input("Tickers", value="AAPL, MSFT, NVDA, SPY, TSLA")
    period = st.selectbox("History Period", options=["1y", "2y", "5y", "10y"], index=2)
    threshold = st.slider("Trade Signal Threshold", min_value=0.50, max_value=0.75, value=0.55, step=0.01)
    forecast_days = st.slider("Projection Days", min_value=5, max_value=60, value=20, step=5)
    n_sims = st.slider("Simulation Paths", min_value=50, max_value=500, value=200, step=50)
    selected_ticker = st.text_input("Focus Ticker", value="AAPL")
    run = st.button("Run Dashboard", use_container_width=True)

if run:
    tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
    with st.spinner("Running models and building dashboard..."):
        df = screen_tickers(tickers, period=period, threshold=threshold)

    st.subheader("Ticker Ranking")
    display_df = df.copy()
    for col in ["Latest Close", "Support", "Resistance"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")
    for col in ["Up Probability", "Holdout Accuracy", "Strategy Return", "Buy & Hold Return"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
    st.dataframe(display_df, use_container_width=True)

    focus = selected_ticker.strip().upper() if selected_ticker.strip() else None
    if focus not in tickers and len(tickers) > 0:
        focus = tickers[0]

    result = train_predict_for_ticker(focus, period=period, threshold=threshold)
    hist = result.history.copy()
    summary, path_df = generate_projection_chart_data(
        result=result,
        forecast_days=forecast_days,
        n_sims=n_sims,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Signal", signal_box(result.model_signal))
    c2.metric("Up Probability", f"{result.next_day_up_probability:.2%}")
    c3.metric("Holdout Accuracy", f"{result.holdout_accuracy:.2%}")
    c4.metric("Strategy Return", f"{result.strategy_return:.2%}")
    c5.metric("Buy & Hold", f"{result.buy_hold_return:.2%}")

    l1, l2, l3, l4, l5 = st.columns(5)
    l1.metric("Latest Close", f"${result.latest_close:,.2f}")
    l2.metric("Support", f"${result.support_level:,.2f}")
    l3.metric("Resistance", f"${result.resistance_level:,.2f}")
    l4.metric("Stop Loss", f"${result.stop_loss:,.2f}")
    l5.metric("Target 1 / 2", f"${result.target_1:,.2f} / ${result.target_2:,.2f}")

    st.plotly_chart(build_candlestick_chart(hist, result), use_container_width=True)
    st.plotly_chart(build_projection_chart(summary), use_container_width=True)

    st.subheader("Projected Price Levels")
    proj = summary.copy()
    for col in proj.columns:
        proj[col] = proj[col].map(lambda x: f"${x:,.2f}")
    st.dataframe(proj, use_container_width=True)

    st.subheader("Feature Importance")
    feature_df = pd.DataFrame(result.top_features, columns=["Feature", "Importance"])
    st.dataframe(feature_df, use_container_width=True)

    st.caption(
        "Signals, targets, support/resistance, and projection charts are model-driven research aids. "
        "They are not guaranteed forecasts or financial advice."
    )
