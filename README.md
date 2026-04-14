# Stock Predictor Movers Fix

This version fixes the top movers strip by:
- using a more reliable public S&P 500 symbol source
- downloading quotes in chunks instead of one giant request
- handling Streamlit/yfinance failures more safely

## Deploy
Replace the repo files with these updated versions and let Streamlit redeploy.
