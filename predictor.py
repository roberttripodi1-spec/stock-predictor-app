from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_10d",
    "sma_5",
    "sma_10",
    "sma_20",
    "ema_12",
    "ema_26",
    "volatility_10",
    "volatility_20",
    "rsi_14",
    "macd",
    "signal",
    "price_vs_sma_20",
    "volume_change",
    "hl_range",
    "oc_change",
    "Volume",
]

POSITIVE_WORDS = {
    "beat", "beats", "surge", "surges", "jump", "jumps", "gain", "gains", "bullish",
    "upgrade", "upgrades", "strong", "growth", "record", "buyback", "expands",
    "momentum", "profit", "profits", "optimistic", "rally", "rallies", "outperform",
    "partnership", "launch", "raises", "raised", "improves", "improved"
}
NEGATIVE_WORDS = {
    "miss", "misses", "drop", "drops", "fall", "falls", "weak", "cut", "cuts",
    "downgrade", "downgrades", "bearish", "lawsuit", "probe", "decline", "declines",
    "warning", "warns", "risk", "risks", "loss", "losses", "recall", "delay", "delays",
    "selloff", "sell-off", "pressure", "slowdown", "recession"
}


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] for c in out.columns]
    return out


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_columns(df)

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out["return_1d"] = out["Close"].pct_change()
    out["return_5d"] = out["Close"].pct_change(5)
    out["return_10d"] = out["Close"].pct_change(10)

    out["sma_5"] = out["Close"].rolling(5).mean()
    out["sma_10"] = out["Close"].rolling(10).mean()
    out["sma_20"] = out["Close"].rolling(20).mean()
    out["sma_50"] = out["Close"].rolling(50).mean()
    out["ema_12"] = out["Close"].ewm(span=12, adjust=False).mean()
    out["ema_26"] = out["Close"].ewm(span=26, adjust=False).mean()

    out["volatility_10"] = out["return_1d"].rolling(10).std()
    out["volatility_20"] = out["return_1d"].rolling(20).std()

    out["rsi_14"] = compute_rsi(out["Close"], 14)
    out["macd"] = out["ema_12"] - out["ema_26"]
    out["signal"] = out["macd"].ewm(span=9, adjust=False).mean()

    out["price_vs_sma_20"] = out["Close"] / out["sma_20"]
    out["volume_change"] = out["Volume"].pct_change()
    out["hl_range"] = (out["High"] - out["Low"]) / out["Close"]
    out["oc_change"] = (out["Close"] - out["Open"]) / out["Open"]

    out["target"] = (out["Close"].shift(-1) > out["Close"]).astype(int)
    out = out.dropna().copy()
    return out


def download_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    data = yf.download(ticker.upper().strip(), period=period, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(f"No data returned for {ticker}")
    return normalize_columns(data)


def train_test_split_time(data: pd.DataFrame, train_size: float = 0.8):
    split_idx = int(len(data) * train_size)
    x = data[FEATURE_COLUMNS]
    y = data["target"]
    return x.iloc[:split_idx], x.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]


def derive_signal(prob: float, price: float, sma20: float, rsi: float, macd: float, macd_signal: float) -> str:
    if prob >= 0.60 and price >= sma20 and rsi < 72 and macd >= macd_signal:
        return "BUY"
    if prob <= 0.40 and price < sma20 and macd < macd_signal:
        return "SELL"
    return "WATCH"


def generate_trade_levels(history: pd.DataFrame, latest_close: float) -> dict:
    recent = history.tail(30).copy()
    support = float(recent["Low"].min())
    resistance = float(recent["High"].max())
    atr_proxy = float((recent["High"] - recent["Low"]).rolling(14).mean().dropna().iloc[-1]) if len(recent) >= 14 else float((recent["High"] - recent["Low"]).mean())

    stop_loss = max(0.01, latest_close - atr_proxy * 1.25)
    target_1 = latest_close + atr_proxy * 1.0
    target_2 = latest_close + atr_proxy * 2.0
    risk = max(0.01, latest_close - stop_loss)
    reward_1 = max(0.0, target_1 - latest_close)
    reward_2 = max(0.0, target_2 - latest_close)

    return {
        "support_level": support,
        "resistance_level": resistance,
        "stop_loss": stop_loss,
        "target_1": target_1,
        "target_2": target_2,
        "rr_1": reward_1 / risk if risk else np.nan,
        "rr_2": reward_2 / risk if risk else np.nan,
    }


def market_mood(prob: float, rsi: float, price: float, sma20: float, sma50: float) -> str:
    score = 0
    if prob > 0.55:
        score += 1
    if rsi > 55:
        score += 1
    if price > sma20:
        score += 1
    if price > sma50:
        score += 1
    if score >= 4:
        return "Bullish"
    if score >= 2:
        return "Neutral"
    return "Bearish"


def _safe_to_datetime(value):
    if value is None:
        return None
    try:
        ts = pd.to_datetime(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts
    except Exception:
        return None


def get_earnings_info(ticker: str) -> dict:
    out = {
        "earnings_date": None,
        "days_to_earnings": None,
        "earnings_flag": "No date found",
    }
    try:
        tk = yf.Ticker(ticker)
        possible = []

        cal = tk.calendar
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            for value in cal.values.flatten():
                possible.append(value)
        elif isinstance(cal, dict):
            for value in cal.values():
                possible.append(value)

        parsed = None
        for value in possible:
            parsed = _safe_to_datetime(value)
            if parsed is not None:
                break

        if parsed is not None:
            now = pd.Timestamp.now(tz="UTC")
            delta = (parsed.normalize() - now.normalize()).days
            out["earnings_date"] = parsed.date().isoformat()
            out["days_to_earnings"] = int(delta)
            if delta < 0:
                out["earnings_flag"] = "Recent earnings"
            elif delta <= 7:
                out["earnings_flag"] = "Earnings soon"
            elif delta <= 30:
                out["earnings_flag"] = "Upcoming earnings"
            else:
                out["earnings_flag"] = "Earnings later"
    except Exception:
        pass

    return out


def get_news_sentiment(ticker: str, max_items: int = 8) -> dict:
    result = {
        "sentiment_score": 0.0,
        "sentiment_label": "Neutral",
        "headline_count": 0,
        "headlines": [],
    }
    try:
        tk = yf.Ticker(ticker)
        news_items = getattr(tk, "news", []) or []
        selected = news_items[:max_items]

        score = 0
        headlines = []
        for item in selected:
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            lower = title.lower()
            pos_hits = sum(1 for w in POSITIVE_WORDS if w in lower)
            neg_hits = sum(1 for w in NEGATIVE_WORDS if w in lower)
            score += (pos_hits - neg_hits)
            headlines.append(title)

        count = len(headlines)
        avg = float(score / count) if count else 0.0

        label = "Neutral"
        if avg >= 0.35:
            label = "Positive"
        elif avg <= -0.35:
            label = "Negative"

        result.update({
            "sentiment_score": avg,
            "sentiment_label": label,
            "headline_count": count,
            "headlines": headlines,
        })
    except Exception:
        pass

    return result


def build_watchlist_flags(
    signal: str,
    mood: str,
    volume_ratio: float,
    earnings_days: int | None,
    sentiment_label: str,
    range_position: float,
    rsi: float,
) -> list[str]:
    flags = []

    if signal == "BUY" and mood == "Bullish":
        flags.append("Trend setup aligned")
    if volume_ratio >= 1.5:
        flags.append("Volume expansion")
    if earnings_days is not None and 0 <= earnings_days <= 7:
        flags.append("Earnings risk this week")
    if sentiment_label == "Positive":
        flags.append("Positive headline tone")
    if sentiment_label == "Negative":
        flags.append("Negative headline tone")
    if range_position >= 0.85:
        flags.append("Near 52-week highs")
    if range_position <= 0.20:
        flags.append("Near 52-week lows")
    if rsi >= 70:
        flags.append("Momentum overheated")
    if rsi <= 35:
        flags.append("Momentum washed out")

    if not flags:
        flags.append("No major alert flags")
    return flags


@dataclass
class ModelResult:
    ticker: str
    rows_used: int
    holdout_accuracy: float
    next_day_up_probability: float
    latest_close: float
    latest_date: str
    top_features: list[tuple[str, float]]
    strategy_return: float
    buy_hold_return: float
    history: pd.DataFrame
    features_data: pd.DataFrame
    model_signal: str
    support_level: float
    resistance_level: float
    stop_loss: float
    target_1: float
    target_2: float
    rr_1: float
    rr_2: float
    rsi_14: float
    macd: float
    macd_signal: float
    sma20: float
    sma50: float
    volume_ratio: float
    momentum_20d: float
    volatility_20: float
    range_52w_position: float
    mood: str
    earnings_date: str | None
    days_to_earnings: int | None
    earnings_flag: str
    sentiment_score: float
    sentiment_label: str
    headline_count: int
    headlines: list[str]
    watchlist_flags: list[str]


def train_predict_for_ticker(ticker: str, period: str = "5y", threshold: float = 0.55) -> ModelResult:
    raw = download_history(ticker, period=period)
    hist = raw.copy()
    data = prepare_features(raw)

    if len(data) < 200:
        raise ValueError(f"Not enough history for {ticker}")

    x_train, x_test, y_train, y_test = train_test_split_time(data)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    probs = model.predict_proba(x_test)[:, 1]
    acc = accuracy_score(y_test, preds)

    latest = data.iloc[[-1]]
    next_up_prob = float(model.predict_proba(latest[FEATURE_COLUMNS])[:, 1][0])

    test_slice = data.iloc[len(x_train):].copy()
    test_slice["up_prob"] = probs
    test_slice["signal"] = (test_slice["up_prob"] >= threshold).astype(int)
    test_slice["market_return"] = test_slice["Close"].pct_change().fillna(0.0)
    test_slice["strategy_return"] = test_slice["signal"].shift(1).fillna(0) * test_slice["market_return"]

    strategy_total = float((1 + test_slice["strategy_return"]).prod() - 1)
    buy_hold_total = float((1 + test_slice["market_return"]).prod() - 1)

    importances = (
        pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
        .sort_values(ascending=False)
        .head(8)
    )

    latest_close = float(data["Close"].iloc[-1])
    latest_sma20 = float(data["sma_20"].iloc[-1])
    latest_sma50 = float(data["sma_50"].iloc[-1]) if "sma_50" in data.columns else latest_sma20
    latest_rsi = float(data["rsi_14"].iloc[-1])
    latest_macd = float(data["macd"].iloc[-1])
    latest_macd_signal = float(data["signal"].iloc[-1])
    volume_ratio = float(hist["Volume"].iloc[-1] / hist["Volume"].tail(20).mean()) if hist["Volume"].tail(20).mean() else np.nan
    momentum_20d = float(hist["Close"].pct_change(20).iloc[-1])
    volatility_20 = float(data["volatility_20"].iloc[-1])

    trailing_252 = hist.tail(min(252, len(hist)))
    range_low = float(trailing_252["Low"].min())
    range_high = float(trailing_252["High"].max())
    range_52w_position = (latest_close - range_low) / max(0.01, (range_high - range_low))

    model_signal = derive_signal(next_up_prob, latest_close, latest_sma20, latest_rsi, latest_macd, latest_macd_signal)
    levels = generate_trade_levels(hist, latest_close)
    mood = market_mood(next_up_prob, latest_rsi, latest_close, latest_sma20, latest_sma50)
    earnings = get_earnings_info(ticker)
    news = get_news_sentiment(ticker)
    flags = build_watchlist_flags(
        signal=model_signal,
        mood=mood,
        volume_ratio=volume_ratio,
        earnings_days=earnings["days_to_earnings"],
        sentiment_label=news["sentiment_label"],
        range_position=float(range_52w_position),
        rsi=latest_rsi,
    )

    return ModelResult(
        ticker=ticker.upper(),
        rows_used=len(data),
        holdout_accuracy=float(acc),
        next_day_up_probability=next_up_prob,
        latest_close=latest_close,
        latest_date=str(data.index[-1].date()),
        top_features=[(k, float(v)) for k, v in importances.items()],
        strategy_return=strategy_total,
        buy_hold_return=buy_hold_total,
        history=hist,
        features_data=data,
        model_signal=model_signal,
        support_level=levels["support_level"],
        resistance_level=levels["resistance_level"],
        stop_loss=levels["stop_loss"],
        target_1=levels["target_1"],
        target_2=levels["target_2"],
        rr_1=levels["rr_1"],
        rr_2=levels["rr_2"],
        rsi_14=latest_rsi,
        macd=latest_macd,
        macd_signal=latest_macd_signal,
        sma20=latest_sma20,
        sma50=latest_sma50,
        volume_ratio=volume_ratio,
        momentum_20d=momentum_20d,
        volatility_20=volatility_20,
        range_52w_position=float(range_52w_position),
        mood=mood,
        earnings_date=earnings["earnings_date"],
        days_to_earnings=earnings["days_to_earnings"],
        earnings_flag=earnings["earnings_flag"],
        sentiment_score=news["sentiment_score"],
        sentiment_label=news["sentiment_label"],
        headline_count=news["headline_count"],
        headlines=news["headlines"],
        watchlist_flags=flags,
    )


def screen_tickers(tickers: Iterable[str], period: str = "5y", threshold: float = 0.55) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        t = ticker.strip().upper()
        if not t:
            continue
        try:
            result = train_predict_for_ticker(t, period=period, threshold=threshold)
            rows.append({
                "Ticker": result.ticker,
                "Signal": result.model_signal,
                "Mood": result.mood,
                "News Sentiment": result.sentiment_label,
                "Earnings": result.earnings_flag,
                "Latest Date": result.latest_date,
                "Latest Close": result.latest_close,
                "Up Probability": result.next_day_up_probability,
                "Holdout Accuracy": result.holdout_accuracy,
                "RSI": result.rsi_14,
                "20D Momentum": result.momentum_20d,
                "Volume Ratio": result.volume_ratio,
                "Strategy Return": result.strategy_return,
                "Buy & Hold Return": result.buy_hold_return,
            })
        except Exception as exc:
            rows.append({
                "Ticker": t,
                "Signal": "ERROR",
                "Mood": "",
                "News Sentiment": "",
                "Earnings": "",
                "Latest Date": "",
                "Latest Close": np.nan,
                "Up Probability": np.nan,
                "Holdout Accuracy": np.nan,
                "RSI": np.nan,
                "20D Momentum": np.nan,
                "Volume Ratio": np.nan,
                "Strategy Return": np.nan,
                "Buy & Hold Return": np.nan,
                "Error": str(exc),
            })

    df = pd.DataFrame(rows)
    if "Up Probability" in df.columns:
        df = df.sort_values(by="Up Probability", ascending=False, na_position="last")
    return df.reset_index(drop=True)


def generate_projection_chart_data(
    result: ModelResult,
    forecast_days: int = 20,
    n_sims: int = 200,
    lookback_days: int = 60,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    hist = result.history.copy()
    close = hist["Close"].dropna().copy()

    returns = close.pct_change().dropna()
    recent_returns = returns.tail(lookback_days)
    if len(recent_returns) < 20:
        recent_returns = returns.tail(min(len(returns), 60))

    base_mu = float(recent_returns.mean())
    sigma = float(recent_returns.std())
    sigma = max(sigma, 0.0001)

    tilt = (result.next_day_up_probability - 0.5) * sigma * 0.5
    drift = base_mu + tilt

    last_price = float(close.iloc[-1])
    future_index = pd.bdate_range(start=close.index[-1] + pd.Timedelta(days=1), periods=forecast_days)

    paths = np.zeros((forecast_days, n_sims))
    for s in range(n_sims):
        price = last_price
        for day in range(forecast_days):
            shock = rng.normal(drift, sigma)
            price = max(0.01, price * (1 + shock))
            paths[day, s] = price

    path_df = pd.DataFrame(paths, index=future_index, columns=[f"path_{i+1}" for i in range(n_sims)])

    summary = pd.DataFrame(index=future_index)
    summary["Median"] = path_df.median(axis=1)
    summary["Low Band (10%)"] = path_df.quantile(0.10, axis=1)
    summary["High Band (90%)"] = path_df.quantile(0.90, axis=1)
    summary["Bull Case (95%)"] = path_df.quantile(0.95, axis=1)
    summary["Bear Case (5%)"] = path_df.quantile(0.05, axis=1)

    return summary, path_df
