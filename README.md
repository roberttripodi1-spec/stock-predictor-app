# Stock Predictor Feature Sanitization Fix

This version fixes the training crash by:
- replacing inf and -inf values in engineered features
- dropping remaining invalid rows before model training
- validating that enough clean rows remain before fitting

If the app hit a ValueError during model.fit, this is the fix for it.
