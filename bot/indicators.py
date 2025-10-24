"""Computation of technical indicators for the trading bot."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import talib

LOGGER = logging.getLogger(__name__)


@dataclass
class IndicatorResult:
    """Container for indicator outputs."""

    frame: pd.DataFrame
    normalized: pd.DataFrame
    signals: pd.DataFrame


def _normalize_columns(frame: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    normalized = frame.copy()
    for column in columns:
        series = frame[column]
        min_val = series.min(skipna=True)
        max_val = series.max(skipna=True)
        if max_val - min_val == 0 or np.isnan(max_val - min_val):
            normalized[column] = 0.0
        else:
            normalized[column] = (series - min_val) / (max_val - min_val)
        normalized[column] = normalized[column].fillna(0.0)
    return normalized


def _detect_rsi_divergence(frame: pd.DataFrame, lookback: int = 14) -> pd.Series:
    """Detect bullish/bearish divergences between price and RSI."""

    price = frame["close"]
    rsi = frame["rsi"]
    divergence = pd.Series(data=0, index=frame.index, dtype=int)

    for i in range(lookback, len(frame)):
        price_slice = price.iloc[i - lookback : i]
        rsi_slice = rsi.iloc[i - lookback : i]

        price_valid = price_slice.dropna()
        rsi_valid = rsi_slice.dropna()
        latest_price = price_slice.iloc[-1]
        latest_rsi = rsi_slice.iloc[-1]

        if price_valid.empty or rsi_valid.empty:
            continue

        if pd.isna(latest_price) or pd.isna(latest_rsi):
            continue

        price_high = price_valid.idxmax()
        price_low = price_valid.idxmin()
        rsi_high = rsi_valid.idxmax()
        rsi_low = rsi_valid.idxmin()

        if latest_price > price_slice.loc[price_high] and latest_rsi < rsi_slice.loc[rsi_high]:
            divergence.iloc[i] = -1  # bearish divergence
        elif latest_price < price_slice.loc[price_low] and latest_rsi > rsi_slice.loc[rsi_low]:
            divergence.iloc[i] = 1  # bullish divergence

    return divergence


def _compute_supertrend(frame: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.Series:
    """Compute SuperTrend direction (+1/-1)."""

    hl2 = (frame["high"] + frame["low"]) / 2.0
    atr = talib.ATR(frame["high"], frame["low"], frame["close"], timeperiod=period)

    basic_upperband = hl2 + multiplier * atr
    basic_lowerband = hl2 - multiplier * atr

    final_upperband = basic_upperband.copy()
    final_lowerband = basic_lowerband.copy()

    for i in range(1, len(frame)):
        final_upperband.iloc[i] = min(basic_upperband.iloc[i], final_upperband.iloc[i - 1]) if frame["close"].iloc[i - 1] > final_upperband.iloc[i - 1] else basic_upperband.iloc[i]
        final_lowerband.iloc[i] = max(basic_lowerband.iloc[i], final_lowerband.iloc[i - 1]) if frame["close"].iloc[i - 1] < final_lowerband.iloc[i - 1] else basic_lowerband.iloc[i]

    supertrend = pd.Series(index=frame.index, dtype=float)
    direction = np.ones(len(frame))

    for i in range(len(frame)):
        if np.isnan(atr.iloc[i]):
            supertrend.iloc[i] = np.nan
            direction[i] = 0
            continue

        if i == 0:
            supertrend.iloc[i] = final_lowerband.iloc[i]
            direction[i] = 1
            continue

        if supertrend.iloc[i - 1] == final_upperband.iloc[i - 1]:
            if frame["close"].iloc[i] <= final_upperband.iloc[i]:
                supertrend.iloc[i] = final_upperband.iloc[i]
                direction[i] = -1
            else:
                supertrend.iloc[i] = final_lowerband.iloc[i]
                direction[i] = 1
        else:
            if frame["close"].iloc[i] >= final_lowerband.iloc[i]:
                supertrend.iloc[i] = final_lowerband.iloc[i]
                direction[i] = 1
            else:
                supertrend.iloc[i] = final_upperband.iloc[i]
                direction[i] = -1

    return pd.Series(direction, index=frame.index, name="supertrend")


def compute_indicators(frame_5m: pd.DataFrame, frame_15m: pd.DataFrame) -> IndicatorResult:
    """Compute multi-timeframe indicators and signals."""

    frame = frame_5m.copy()
    frame.sort_values("open_time", inplace=True)
    frame.set_index("open_time", inplace=True)

    frame["rsi"] = talib.RSI(frame["close"], timeperiod=14)
    frame["macd"], frame["macd_signal"], frame["macd_hist"] = talib.MACD(
        frame["close"], fastperiod=12, slowperiod=26, signalperiod=9
    )
    frame["ema_9"] = talib.EMA(frame["close"], timeperiod=9)
    frame["ema_21"] = talib.EMA(frame["close"], timeperiod=21)
    frame["ema_50"] = talib.EMA(frame["close"], timeperiod=50)
    frame["ema_200"] = talib.EMA(frame["close"], timeperiod=200)
    frame["adx"] = talib.ADX(frame["high"], frame["low"], frame["close"], timeperiod=14)

    upper, middle, lower = talib.BBANDS(frame["close"], timeperiod=20, nbdevup=2, nbdevdn=2)
    frame["bb_upper"] = upper
    frame["bb_middle"] = middle
    frame["bb_lower"] = lower
    frame["bb_width"] = (upper - lower) / middle
    frame["bb_squeeze"] = np.where(frame["bb_width"] < frame["bb_width"].rolling(window=20).quantile(0.2), 1, 0)

    frame["supertrend"] = _compute_supertrend(frame, period=10, multiplier=3.0)
    frame["rsi_divergence"] = _detect_rsi_divergence(frame)

    frame["adx_filter"] = np.where(frame["adx"] > 25, 1, 0)
    frame["macd_signal_cross"] = np.where(frame["macd"] > frame["macd_signal"], 1, -1)
    frame["ema_trend"] = np.where(frame["ema_9"] > frame["ema_21"], 1, -1)

    frame["bollinger_breakout"] = np.select(
        [frame["close"] > frame["bb_upper"], frame["close"] < frame["bb_lower"]],
        [1, -1],
        default=0,
    )

    frame["candle_hammer"] = talib.CDLHAMMER(frame["open"], frame["high"], frame["low"], frame["close"])
    frame["candle_doji"] = talib.CDLDOJI(frame["open"], frame["high"], frame["low"], frame["close"])
    frame["candle_engulfing"] = talib.CDLENGULFING(frame["open"], frame["high"], frame["low"], frame["close"])
    frame["candle_morning_star"] = talib.CDLMORNINGSTAR(frame["open"], frame["high"], frame["low"], frame["close"])
    frame["candle_evening_star"] = talib.CDLEVENINGSTAR(frame["open"], frame["high"], frame["low"], frame["close"])

    frame["rsi_divergence_signal"] = frame["rsi_divergence"]
    frame["macd_hist_signal"] = np.sign(frame["macd_hist"])
    frame["supertrend_signal"] = frame["supertrend"]
    frame["ema_cross_signal"] = np.where(frame["ema_9"] > frame["ema_21"], 1, -1)
    frame["adx_signal"] = frame["adx_filter"]

    # 15 minute EMA trend for confirmation
    frame_15m = frame_15m.copy()
    frame_15m.sort_values("open_time", inplace=True)
    frame_15m.set_index("open_time", inplace=True)
    frame_15m["ema_50"] = talib.EMA(frame_15m["close"], timeperiod=50)
    frame_15m["ema_200"] = talib.EMA(frame_15m["close"], timeperiod=200)
    frame_15m["higher_tf_trend"] = np.where(frame_15m["ema_50"] > frame_15m["ema_200"], 1, -1)
    frame = frame.join(frame_15m[["higher_tf_trend"]], how="left")
    frame["higher_tf_trend"] = frame["higher_tf_trend"].ffill().fillna(0)

    signal_columns = [
        "rsi_divergence_signal",
        "macd_hist_signal",
        "supertrend_signal",
        "ema_cross_signal",
        "adx_signal",
        "bollinger_breakout",
        "higher_tf_trend",
    ]

    indicator_columns = [
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
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "bb_width",
    ]

    normalized = _normalize_columns(frame[indicator_columns], indicator_columns)
    signals = frame[signal_columns].fillna(0).astype(float)

    return IndicatorResult(frame=frame, normalized=normalized, signals=signals)
