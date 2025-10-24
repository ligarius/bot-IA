import os
import sys

import inspect

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bot.indicators import (
    IndicatorResult,
    _detect_rsi_divergence,
    compute_indicators,
)


def _build_frame(start, periods, freq):
    index = pd.date_range(start=start, periods=periods, freq=freq)
    data = {
        "open_time": index,
        "open": np.linspace(100, 100 + periods - 1, periods),
        "high": np.linspace(101, 101 + periods - 1, periods),
        "low": np.linspace(99, 99 + periods - 1, periods),
        "close": np.linspace(100, 100 + periods - 1, periods),
        "volume": np.linspace(1, periods, periods),
    }
    return pd.DataFrame(data)


def test_compute_indicators_handles_nan_rsi():
    lookback = inspect.signature(_detect_rsi_divergence).parameters["lookback"].default

    frame_5m = _build_frame("2024-01-01", periods=lookback + 6, freq="5min")
    frame_15m = _build_frame("2024-01-01", periods=4, freq="15min")

    frame_5m.loc[frame_5m.index[-(lookback + 1) :], "close"] = np.nan

    result = compute_indicators(frame_5m, frame_15m)

    assert isinstance(result, IndicatorResult)
    assert (result.signals["rsi_divergence_signal"] == 0).all()
