"""Backtesting utilities for the trading bot."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from .config import CONFIG
from .features import LSTM_FEATURE_COLUMNS
from .indicators import compute_indicators
from .strategy import EnsembleStrategy, StrategyDecision
from .trader import Trader

LOGGER = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    timestamp: pd.Timestamp
    action: str
    price: float
    reason: str


@dataclass
class BacktestResult:
    trades: List[TradeRecord] = field(default_factory=list)
    pnl: float = 0.0
    win_rate: float = 0.0
    total_return: float = 0.0
    equity_curve: List[float] = field(default_factory=list)


class Backtester:
    """Run backtests using the ensemble strategy and trader simulation."""

    def __init__(self, strategy: EnsembleStrategy, trader: Trader) -> None:
        self.strategy = strategy
        self.trader = trader

    def run(
        self,
        frame_5m: pd.DataFrame,
        frame_15m: pd.DataFrame,
        lstm_predictions: np.ndarray | None = None,
    ) -> BacktestResult:
        indicators = compute_indicators(frame_5m, frame_15m)
        normalized = indicators.normalized.dropna()
        feature_frame = indicators.frame[LSTM_FEATURE_COLUMNS].dropna()
        common_index = normalized.index.intersection(feature_frame.index)

        normalized = normalized.reindex(common_index)
        signals = indicators.signals.reindex(common_index).fillna(0)
        price_series = frame_5m.set_index("open_time")["close"].reindex(common_index)

        if lstm_predictions is not None:
            series_length = len(common_index)
            if series_length == 0:
                LOGGER.warning(
                    "No aligned data available for LSTM predictions; ignoring predictions in backtest"
                )
                lstm_predictions = None
            else:
                lstm_predictions = np.asarray(lstm_predictions)
                lstm_predictions = lstm_predictions[-series_length:]
                if len(lstm_predictions) != series_length:
                    LOGGER.warning(
                        "LSTM predictions length mismatch after alignment; ignoring predictions in backtest"
                    )
                    lstm_predictions = None

        trades: List[TradeRecord] = []
        equity = CONFIG.initial_capital
        equity_curve: List[float] = [equity]
        wins = 0
        losses = 0

        for idx in normalized.index:
            row = normalized.loc[idx]
            signal_row = signals.loc[idx]
            price = price_series.loc[idx]

            lstm_vector = None
            if lstm_predictions is not None and len(lstm_predictions) == len(normalized):
                lstm_vector = lstm_predictions[normalized.index.get_loc(idx)]

            decision: StrategyDecision = self.strategy.decide(row, signal_row, lstm_vector)

            if self.trader.position.is_open():
                exited, pnl_value = self.trader.evaluate_exit(price)
                if exited:
                    pnl_pct = pnl_value / CONFIG.trade_size if CONFIG.trade_size else 0
                    equity *= 1 + pnl_pct
                    if pnl_value > 0:
                        wins += 1
                    else:
                        losses += 1
                    trades.append(
                        TradeRecord(
                            timestamp=idx,
                            action="EXIT",
                            price=price,
                            reason="Risk management exit",
                        )
                    )

            if decision.action == 1:
                self.trader.open_position(price)
                trades.append(TradeRecord(timestamp=idx, action="BUY", price=price, reason=decision.reason))
            elif decision.action == 2:
                if self.trader.position.is_open():
                    pnl_value = self.trader.close_position(price)
                    pnl = pnl_value / CONFIG.trade_size if CONFIG.trade_size else 0
                    if pnl_value > 0:
                        wins += 1
                    else:
                        losses += 1
                    equity *= 1 + pnl
                    trades.append(TradeRecord(timestamp=idx, action="SELL", price=price, reason=decision.reason))

            equity_curve.append(equity)

        total_trades = wins + losses
        win_rate = wins / total_trades if total_trades > 0 else 0.0
        total_return = (equity - CONFIG.initial_capital) / CONFIG.initial_capital

        return BacktestResult(
            trades=trades,
            pnl=equity - CONFIG.initial_capital,
            win_rate=win_rate,
            total_return=total_return,
            equity_curve=equity_curve,
        )
