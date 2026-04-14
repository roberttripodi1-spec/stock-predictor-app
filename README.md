# Stock Predictor Clean V10

This file set is a cleaner drop-in replacement for the previous Streamlit app.

## What changed
- removed the fixed bottom navigation
- removed the CSS hamburger override that was causing duplicate/ugly menu behavior
- kept the native Streamlit sidebar toggle
- added a compact top page nav instead of bottom nav
- tightened mobile spacing across cards and metrics
- fixed the indicator gauges so values sit under the arch
- placed the two gauges in a responsive two-column layout on the details page
- kept the predictor and screener logic intact

## Files
- `app.py` — cleaned UI and navigation
- `predictor.py` — model and projection logic
- `screen_stocks.py` — batch screening utility
- `requirements.txt`

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## GitHub swap
Replace your old repo files with these files, then redeploy.
