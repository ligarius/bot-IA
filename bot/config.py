"""Global configuration and logging utilities for the trading bot."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables as early as possible.
load_dotenv()


def _ensure_log_dir(log_dir: Path) -> None:
    """Ensure the logging directory exists."""
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to create log directory {log_dir}") from exc


@dataclass(frozen=True)
class BinanceCredentials:
    """Container for Binance API credentials."""

    api_key: str
    api_secret: str

    @classmethod
    def from_env(cls) -> "BinanceCredentials":
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        if not api_key or not api_secret:
            raise RuntimeError(
                "Missing Binance API credentials in environment variables."
            )
        return cls(api_key=api_key, api_secret=api_secret)


@dataclass
class BotConfig:
    """Core configuration for the trading system."""

    symbol: str = "BTCUSDT"
    base_asset: str = "BTC"
    quote_asset: str = "USDT"
    interval_live: str = "5m"
    interval_backtest: str = "5m"
    lstm_sequence_length: int = 60
    rl_window: int = 20
    initial_capital: float = 90.0
    trade_size: float = 10.0
    stop_loss_pct: float = 0.015
    take_profit_pct: float = 0.03
    fee_pct: float = 0.001
    data_points_short: int = 500
    data_points_long: int = 500
    model_dir: Path = Path("models")
    log_dir: Path = Path("logs")
    cache_dir: Path = Path("cache")

    def ensure_directories(self) -> None:
        for directory in (self.model_dir, self.log_dir, self.cache_dir):
            _ensure_log_dir(directory)


CONFIG = BotConfig()
CONFIG.ensure_directories()


def configure_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """Configure the root logger with a rotating file handler."""

    logger = logging.getLogger()
    if logger.handlers:
        return

    log_path = CONFIG.log_dir / (log_file or "bot.log")
    handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=5)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console_handler)


configure_logging()
