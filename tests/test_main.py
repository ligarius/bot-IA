"""Tests for bot.main utility functions."""

from unittest.mock import MagicMock

from bot import main
from bot.config import CONFIG


def test_load_data_uses_interval_minutes() -> None:
    loader = MagicMock()
    loader.load_klines.return_value = "sentinel"

    samples = 12
    result = main._load_data(loader, "15m", samples)

    assert result == "sentinel"
    loader.load_klines.assert_called_once()

    args, kwargs = loader.load_klines.call_args
    assert args == (CONFIG.symbol, "15m")
    assert kwargs["lookback"] == f"{samples * 15} minutes ago UTC"
    assert kwargs["min_samples"] == samples
