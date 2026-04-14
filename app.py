from __future__ import annotations

import streamlit as st
import pandas as pd

from predictor import screen_tickers, train_predict_for_ticker

st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("Stock Predictor Dashboard")
st.caption("Simple next-day direction model using daily price data and technical indicators.")

with st.sidebar:
    tickers_text = st.text_input("Tickers", value="AAPL, MSFT, NVDA, SPY, TSLA")
    period = st.selectbox("History Period", options=["1y", "2y", "5y", "10y"], index=2)
    threshold = st.slider("Trade Signal Threshold", min_value=0.50, max_value=0.75, value=0.55, step=0.01)
    run = st.button("Run Screen", use_container_width=True)

if run:
    tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
    with st.spinner("Training models and screening tickers..."):
        df = screen_tickers(tickers, period=period, threshold=threshold)

    st.subheader("Screen Results")
    display_df = df.copy()
    for col in ["Up Probability", "Holdout Accuracy", "Strategy Return", "Buy & Hold Return"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
    st.dataframe(display_df, use_container_width=True)

    good = df[df["Ticker"].notna() & df["Up Probability"].notna()]
    if not good.empty:
        top_ticker = good.iloc[0]["Ticker"]
        st.subheader(f"Top Ranked Ticker: {top_ticker}")
        result = train_predict_for_ticker(top_ticker, period=period, threshold=threshold)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Next-Day Up Probability", f"{result.next_day_up_probability:.2%}")
        c2.metric("Holdout Accuracy", f"{result.holdout_accuracy:.2%}")
        c3.metric("Strategy Return", f"{result.strategy_return:.2%}")
        c4.metric("Buy & Hold Return", f"{result.buy_hold_return:.2%}")

        st.write(f"Latest close: **${result.latest_close:,.2f}** on **{result.latest_date}**")

        st.subheader("Top Feature Importances")
        feature_df = pd.DataFrame(result.top_features, columns=["Feature", "Importance"])
        st.dataframe(feature_df, use_container_width=True)

st.markdown("---")
st.write(
    "This is a research starter, not financial advice. "
    "A model with decent historical accuracy can still fail badly in live trading."
)
