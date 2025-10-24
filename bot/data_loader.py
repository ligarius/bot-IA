"""Utilities for loading historical and live market data."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

from .config import BinanceCredentials, CONFIG

LOGGER = logging.getLogger(__name__)


class MarketDataLoader:
    """Load market data from Binance or local cache."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.cache_dir = cache_dir or CONFIG.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            credentials = BinanceCredentials.from_env()
        except RuntimeError as exc:
            LOGGER.warning("Binance credentials not available: %s", exc)
            credentials = None

        self.client: Optional[Client]
        if credentials is None:
            self.client = None
        else:
            self.client = Client(api_key=credentials.api_key, api_secret=credentials.api_secret)

    def _cache_file(self, symbol: str, interval: str) -> Path:
        return self.cache_dir / f"{symbol}_{interval}.json"

    def load_klines(
        self,
        symbol: str,
        interval: str,
        lookback: str = "2 day ago UTC",
        min_samples: int = 500,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Load historical klines, using cache when available."""

        cache_file = self._cache_file(symbol, interval)
        if use_cache and cache_file.exists():
            try:
                LOGGER.debug("Loading data from cache: %s", cache_file)
                with cache_file.open("r", encoding="utf-8") as fh:
                    raw = json.load(fh)
                frame = self._klines_to_frame(raw)
                if len(frame) >= min_samples:
                    return frame.tail(min_samples)
            except (OSError, json.JSONDecodeError) as exc:
                LOGGER.warning("Failed to read cache %s: %s", cache_file, exc)

        if self.client is None:
            raise RuntimeError(
                "Binance client not available. Provide credentials or disable live fetching."
            )

        try:
            LOGGER.info("Fetching klines for %s %s", symbol, interval)
            raw = self.client.get_historical_klines(symbol, interval, lookback)
        except (BinanceAPIException, BinanceRequestException) as exc:
            raise RuntimeError("Failed to fetch klines from Binance") from exc

        try:
            with cache_file.open("w", encoding="utf-8") as fh:
                json.dump(raw, fh)
        except OSError as exc:
            LOGGER.warning("Failed to write cache %s: %s", cache_file, exc)

        return self._klines_to_frame(raw).tail(min_samples)

    @staticmethod
    def _klines_to_frame(klines: list[list[str]]) -> pd.DataFrame:
        """Convert raw klines into a pandas DataFrame."""

        columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore",
        ]
        frame = pd.DataFrame(klines, columns=columns)
        frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms")
        frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms")
        numeric_cols = ["open", "high", "low", "close", "volume"]
        frame[numeric_cols] = frame[numeric_cols].astype(float)
        return frame

    def stream_live(self, symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
        """Fetch the most recent candles for live trading."""

        return self.load_klines(symbol, interval, lookback=f"{limit} minutes ago UTC", min_samples=limit)


def download_binance_data(symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Fallback downloader using Binance public API without authentication."""

    LOGGER.info("Downloading Binance data via REST API for %s %s", symbol, interval)
    endpoint = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": int(start.timestamp() * 1000),
        "endTime": int(end.timestamp() * 1000),
        "limit": 1000,
    }

    try:
        response = requests.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError("Failed to download Binance data") from exc

    raw = response.json()
    if not raw:
        raise RuntimeError("Empty response from Binance public API")

    frame = MarketDataLoader._klines_to_frame(raw)
    return frame


def load_local_csv(path: os.PathLike[str] | str) -> pd.DataFrame:
    """Load OHLCV data from a CSV file."""

    try:
        frame = pd.read_csv(path, parse_dates=["open_time", "close_time"])
    except (OSError, pd.errors.ParserError) as exc:
        raise RuntimeError(f"Failed to read CSV data at {path}") from exc

    required_columns = {"open", "high", "low", "close", "volume"}
    if not required_columns.issubset(frame.columns):
        raise ValueError(f"CSV file missing required columns: {required_columns}")

    return frame
