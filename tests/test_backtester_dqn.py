"""Integration tests for Backtester handling of DQN-driven actions."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch

if "binance" not in sys.modules:
    binance_module = types.ModuleType("binance")
    client_module = types.ModuleType("binance.client")

    class _Client:  # pragma: no cover - simple stub
        pass

    client_module.Client = _Client

    enums_module = types.ModuleType("binance.enums")
    enums_module.SIDE_BUY = "BUY"
    enums_module.SIDE_SELL = "SELL"
    enums_module.ORDER_TYPE_MARKET = "MARKET"

    exceptions_module = types.ModuleType("binance.exceptions")

    class _BinanceError(Exception):
        pass

    exceptions_module.BinanceAPIException = _BinanceError
    exceptions_module.BinanceRequestException = _BinanceError

    sys.modules["binance"] = binance_module
    sys.modules["binance.client"] = client_module
    sys.modules["binance.enums"] = enums_module
    sys.modules["binance.exceptions"] = exceptions_module

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bot.backtester import Backtester
from bot.config import CONFIG
from bot.indicators import IndicatorResult
from bot.strategy import StrategyDecision, EnsembleStrategy
from bot.features import LSTM_FEATURE_COLUMNS


@dataclass
class _DummyPosition:
    side: str | None = None
    entry_price: float | None = None
    quantity: float = 0.0

    def is_open(self) -> bool:
        return self.side is not None and self.quantity > 0


class _FakeTrader:
    def __init__(self) -> None:
        self.position = _DummyPosition()
        self.trade_size = CONFIG.trade_size

    def open_position(self, price: float) -> None:
        if self.position.is_open():
            return
        qty = self.trade_size / price if price else 0.0
        self.position = _DummyPosition(side="BUY", entry_price=price, quantity=qty)

    def close_position(self, price: float) -> float:
        if not self.position.is_open() or self.position.entry_price is None:
            return 0.0
        qty = self.position.quantity
        entry = self.position.entry_price
        pnl = (price - entry) * qty
        self.position = _DummyPosition()
        return pnl

    def evaluate_exit(self, price: float) -> tuple[bool, float]:  # pragma: no cover - deterministic stub
        return False, 0.0


class _DQNForwardStrategy(EnsembleStrategy):
    """Strategy wrapper that mirrors DQN actions for testing purposes."""

    def decide(self, indicators, signals, lstm_prediction=None, dqn_action=None):  # type: ignore[override]
        action = dqn_action if dqn_action is not None else 0
        return StrategyDecision(action=action, reason=f"DQN action={action}")


def test_backtester_records_trades_when_dqn_forces_positions() -> None:
    index = pd.date_range("2023-01-01", periods=CONFIG.rl_window + 3, freq="5min")
    feature_values = {
        column: np.linspace(0.1, 0.9, len(index), dtype=np.float32)
        for column in LSTM_FEATURE_COLUMNS
    }
    frame = pd.DataFrame(feature_values, index=index)
    normalized = pd.DataFrame({"feature": np.linspace(-1, 1, len(index))}, index=index)
    signals = pd.DataFrame({"placeholder": np.zeros(len(index))}, index=index)

    indicator_result = IndicatorResult(frame=frame, normalized=normalized, signals=signals)

    frame_5m = pd.DataFrame(
        {
            "open_time": index,
            "open": np.linspace(100, 100 + len(index) - 1, len(index)),
            "high": np.linspace(101, 101 + len(index) - 1, len(index)),
            "low": np.linspace(99, 99 + len(index) - 1, len(index)),
            "close": np.linspace(100, 100 + len(index) - 1, len(index)),
            "volume": np.ones(len(index)),
        }
    )
    frame_15m = frame_5m.copy()

    lstm_predictions = np.zeros((len(index), 3), dtype=np.float32)
    actionable_index = index[CONFIG.rl_window]
    exit_index = index[CONFIG.rl_window + 1]
    dqn_actions = {actionable_index: 1, exit_index: 2}

    strategy = _DQNForwardStrategy()
    trader = _FakeTrader()
    backtester = Backtester(strategy, trader)

    with patch("bot.backtester.compute_indicators", return_value=indicator_result):
        result = backtester.run(frame_5m, frame_15m, lstm_predictions, dqn_actions)

    assert any(trade.action == "BUY" for trade in result.trades)
    assert any(trade.action == "SELL" for trade in result.trades)
