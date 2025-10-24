"""Command line interface for the Binance trading bot."""
from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np
import pandas as pd

from .backtester import Backtester
from .config import CONFIG, configure_logging
from .data_loader import MarketDataLoader
from .features import LSTM_FEATURE_COLUMNS
from .indicators import compute_indicators
from .lstm_model import LSTMTrainer
from .rl_agent import RLAgent, TradingEnvironment
from .strategy import EnsembleStrategy
from .trader import Trader

LOGGER = logging.getLogger(__name__)


def _load_data(loader: MarketDataLoader, interval: str, samples: int) -> pd.DataFrame:
    lookback_minutes = samples * 5
    lookback = f"{lookback_minutes} minutes ago UTC"
    return loader.load_klines(CONFIG.symbol, interval, lookback=lookback, min_samples=samples)


def train_lstm(loader: MarketDataLoader) -> None:
    frame_5m = _load_data(loader, "5m", max(CONFIG.data_points_short, CONFIG.lstm_sequence_length + 100))
    frame_15m = _load_data(loader, "15m", CONFIG.data_points_long)
    indicators = compute_indicators(frame_5m, frame_15m)
    features = indicators.frame[LSTM_FEATURE_COLUMNS].dropna()
    trainer = LSTMTrainer()
    metrics = trainer.train(features)
    LOGGER.info("LSTM training metrics: %s", metrics)


def train_rl(loader: MarketDataLoader) -> None:
    trainer = LSTMTrainer()
    model = trainer.load_model()

    frame_5m = _load_data(loader, "5m", CONFIG.data_points_short + CONFIG.rl_window + 10)
    frame_15m = _load_data(loader, "15m", CONFIG.data_points_long)
    indicators = compute_indicators(frame_5m, frame_15m)

    feature_df = indicators.frame[LSTM_FEATURE_COLUMNS].dropna()
    normalized_df = indicators.normalized.dropna()
    common_index = normalized_df.index.intersection(feature_df.index)
    feature_df = feature_df.loc[common_index]
    normalized_df = normalized_df.loc[common_index]

    if not feature_df.index.equals(normalized_df.index):
        raise RuntimeError("Feature and normalized data are not aligned for RL training")

    LOGGER.info("RL training dataset aligned with %d samples", len(common_index))

    min_required = max(CONFIG.lstm_sequence_length, CONFIG.rl_window + 1)
    if len(normalized_df) <= min_required:
        raise RuntimeError("Insufficient data for RL training")

    normalized = normalized_df.to_numpy()
    close_series = frame_5m.set_index("open_time")["close"].loc[normalized_df.index]
    close_prices = close_series.to_numpy()
    lstm_predictions = np.zeros((len(normalized_df), 3))

    for idx in range(CONFIG.lstm_sequence_length, len(normalized_df)):
        window = feature_df.iloc[idx - CONFIG.lstm_sequence_length : idx]
        prediction = trainer.infer(model, window)
        lstm_predictions[idx] = prediction

    price_diff = np.diff(close_prices, prepend=close_prices[0])
    rewards = price_diff / close_prices

    env = TradingEnvironment(
        normalized=normalized,
        rewards=rewards,
        lstm_predictions=lstm_predictions,
        trade_size=CONFIG.trade_size,
    )
    agent = RLAgent()
    agent.train(env)


def run_backtest(loader: MarketDataLoader) -> None:
    frame_5m = _load_data(loader, "5m", CONFIG.data_points_short + 200)
    frame_15m = _load_data(loader, "15m", CONFIG.data_points_long + 200)
    indicators = compute_indicators(frame_5m, frame_15m)

    trainer = LSTMTrainer()
    feature_df = indicators.frame[LSTM_FEATURE_COLUMNS].dropna()
    normalized_df = indicators.normalized.dropna()
    common_index = normalized_df.index.intersection(feature_df.index)
    feature_df = feature_df.loc[common_index]
    normalized_df = normalized_df.loc[common_index]

    if not feature_df.index.equals(normalized_df.index):
        raise RuntimeError("Feature and normalized data are not aligned for backtesting")

    try:
        model = trainer.load_model()
        if len(normalized_df) > CONFIG.lstm_sequence_length:
            lstm_predictions = np.zeros((len(normalized_df), 3))
            for idx in range(CONFIG.lstm_sequence_length, len(normalized_df)):
                window = feature_df.iloc[idx - CONFIG.lstm_sequence_length : idx]
                prediction = trainer.infer(model, window)
                lstm_predictions[idx] = prediction
            LOGGER.info("Backtest LSTM predictions aligned for %d samples (last timestamp: %s)", len(normalized_df), normalized_df.index[-1])
        else:
            lstm_predictions = None
    except FileNotFoundError:
        LOGGER.warning("LSTM model not found, proceeding without predictions")
        lstm_predictions = None

    strategy = EnsembleStrategy()
    trader = Trader(live=False)
    backtester = Backtester(strategy, trader)
    result = backtester.run(frame_5m, frame_15m, lstm_predictions)

    LOGGER.info("Backtest completed | PnL: %.2f | Win rate: %.2f%% | Total return: %.2f%%", result.pnl, result.win_rate * 100, result.total_return * 100)
    for trade in result.trades[-10:]:
        LOGGER.info("Trade: %s %s @ %.2f | %s", trade.timestamp, trade.action, trade.price, trade.reason)


def run_paper_trading(loader: MarketDataLoader) -> None:
    trader = Trader(live=False)
    strategy = EnsembleStrategy()
    trainer = LSTMTrainer()

    try:
        model = trainer.load_model()
    except FileNotFoundError as exc:
        LOGGER.error("Paper trading requires a trained LSTM model: %s", exc)
        return

    frame_15m = _load_data(loader, "15m", CONFIG.data_points_long)

    while True:
        frame_5m = loader.stream_live(CONFIG.symbol, CONFIG.interval_live, limit=CONFIG.lstm_sequence_length + 10)
        indicators = compute_indicators(frame_5m, frame_15m)
        feature_frame = indicators.frame[LSTM_FEATURE_COLUMNS].dropna()
        normalized = indicators.normalized.dropna()
        common_index = normalized.index.intersection(feature_frame.index)
        feature_frame = feature_frame.loc[common_index]
        normalized = normalized.loc[common_index]

        if not feature_frame.index.equals(normalized.index):
            LOGGER.warning("Feature and normalized data misaligned during paper trading iteration")
            continue

        if len(normalized) < CONFIG.lstm_sequence_length:
            LOGGER.warning("Not enough data for decision making yet")
            continue

        window = feature_frame.tail(CONFIG.lstm_sequence_length)
        lstm_prediction = trainer.infer(model, window)

        latest_time = normalized.index[-1]
        signals = indicators.signals.loc[latest_time]
        row = normalized.loc[latest_time]
        price = frame_5m.tail(1)["close"].iloc[0]

        decision = strategy.decide(row, signals, lstm_prediction)
        LOGGER.debug("Paper trading alignment check | time=%s | feature_shape=%s | normalized_shape=%s", latest_time, window.shape, normalized.shape)
        LOGGER.info("Decision at %s: %s", latest_time, decision.reason)

        if decision.action == 1:
            trader.open_position(price)
        elif decision.action == 2:
            trader.close_position(price)
        else:
            trader.evaluate_exit(price)

        time.sleep(60)


def run_live_trading(loader: MarketDataLoader) -> None:
    LOGGER.warning("Live trading mode should be thoroughly tested before deployment.")
    trader = Trader(live=True)
    strategy = EnsembleStrategy()
    trainer = LSTMTrainer()
    model = trainer.load_model()
    frame_15m = _load_data(loader, "15m", CONFIG.data_points_long)

    while True:
        frame_5m = loader.stream_live(CONFIG.symbol, CONFIG.interval_live, limit=CONFIG.lstm_sequence_length + 10)
        indicators = compute_indicators(frame_5m, frame_15m)
        feature_frame = indicators.frame[LSTM_FEATURE_COLUMNS].dropna()
        normalized = indicators.normalized.dropna()
        common_index = normalized.index.intersection(feature_frame.index)
        feature_frame = feature_frame.loc[common_index]
        normalized = normalized.loc[common_index]

        if not feature_frame.index.equals(normalized.index):
            LOGGER.warning("Feature and normalized data misaligned during live trading iteration")
            continue

        if len(normalized) < CONFIG.lstm_sequence_length:
            LOGGER.warning("Not enough data for decision making yet")
            continue

        window = feature_frame.tail(CONFIG.lstm_sequence_length)
        lstm_prediction = trainer.infer(model, window)

        latest_time = normalized.index[-1]
        signals = indicators.signals.loc[latest_time]
        row = normalized.loc[latest_time]
        price = frame_5m.tail(1)["close"].iloc[0]

        decision = strategy.decide(row, signals, lstm_prediction)
        LOGGER.debug(
            "Live trading alignment check | time=%s | feature_shape=%s | normalized_shape=%s",
            latest_time,
            window.shape,
            normalized.shape,
        )
        LOGGER.info("Live decision at %s: %s", latest_time, decision.reason)

        if decision.action == 1:
            trader.open_position(price)
        elif decision.action == 2:
            trader.close_position(price)
        else:
            trader.evaluate_exit(price)

        time.sleep(60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Binance Trading Bot")
    parser.add_argument("--train-lstm", action="store_true", help="Train the LSTM model")
    parser.add_argument("--train-rl", action="store_true", help="Train the DQN agent")
    parser.add_argument("--backtest", action="store_true", help="Run historical backtest")
    parser.add_argument("--paper", action="store_true", help="Run paper trading loop")
    parser.add_argument("--live", action="store_true", help="Run live trading on Binance")

    args = parser.parse_args(argv)
    configure_logging()
    loader = MarketDataLoader()

    try:
        if args.train_lstm:
            train_lstm(loader)
        if args.train_rl:
            train_rl(loader)
        if args.backtest:
            run_backtest(loader)
        if args.paper:
            run_paper_trading(loader)
        if args.live:
            run_live_trading(loader)
        if not any(vars(args).values()):
            parser.print_help()
            return 1
    except Exception as exc:  # pragma: no cover - high-level safeguard
        LOGGER.exception("Application error: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
