"""Shared feature column definitions for model training and inference."""

LSTM_FEATURE_COLUMNS = [
    "close",
    "volume",
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "ema_9",
    "ema_21",
    "ema_50",
    "ema_200",
    "supertrend",
    "adx",
]
