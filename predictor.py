from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
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
    return data


def train_test_split_time(data: pd.DataFrame, train_size: float = 0.8):
    split_idx = int(len(data) * train_size)
    x = data[FEATURE_COLUMNS]
    y = data["target"]
    return x.iloc[:split_idx], x.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]


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


def train_predict_for_ticker(ticker: str, period: str = "5y", threshold: float = 0.55) -> ModelResult:
    raw = download_history(ticker, period=period)
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

    return ModelResult(
        ticker=ticker.upper(),
        rows_used=len(data),
        holdout_accuracy=float(acc),
        next_day_up_probability=next_up_prob,
        latest_close=float(data["Close"].iloc[-1]),
        latest_date=str(data.index[-1].date()),
        top_features=[(k, float(v)) for k, v in importances.items()],
        strategy_return=strategy_total,
        buy_hold_return=buy_hold_total,
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
                "Latest Date": result.latest_date,
                "Latest Close": result.latest_close,
                "Up Probability": result.next_day_up_probability,
                "Holdout Accuracy": result.holdout_accuracy,
                "Strategy Return": result.strategy_return,
                "Buy & Hold Return": result.buy_hold_return,
            })
        except Exception as exc:
            rows.append({
                "Ticker": t,
                "Latest Date": "",
                "Latest Close": np.nan,
                "Up Probability": np.nan,
                "Holdout Accuracy": np.nan,
                "Strategy Return": np.nan,
                "Buy & Hold Return": np.nan,
                "Error": str(exc),
            })

    df = pd.DataFrame(rows)
    if "Up Probability" in df.columns:
        df = df.sort_values(by="Up Probability", ascending=False, na_position="last")
    return df.reset_index(drop=True)


def save_model(ticker: str, period: str = "5y", out_dir: str = "model_output") -> Path:
    raw = download_history(ticker, period=period)
    data = prepare_features(raw)
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

    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    model_path = out / f"{ticker.upper()}_model.joblib"
    joblib.dump(model, model_path)
    return model_path
