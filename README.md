# Stock Predictor v2

This version includes:

- `predictor.py` — reusable core functions
- `app.py` — Streamlit dashboard
- `screen_stocks.py` — CLI screener for multiple tickers
- `requirements.txt` — dependencies

## Install
```bash
pip install -r requirements.txt
```

## Run dashboard
```bash
streamlit run app.py
```

## Run screener
```bash
python screen_stocks.py --tickers AAPL,MSFT,NVDA,SPY,TSLA --period 5y --threshold 0.55
```

## Notes
- Predicts next-day direction only
- Uses technical indicators from daily OHLCV data
- Includes a basic threshold-based backtest on the holdout period
- Intended for research and idea generation, not blind trading
