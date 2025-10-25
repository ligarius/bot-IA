"""Command line interface for the Binance trading bot."""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtester import Backtester, BacktestResult
from .config import CONFIG, configure_logging
from .data_loader import MarketDataLoader
from .features import LSTM_FEATURE_COLUMNS
from .indicators import compute_indicators
from .lstm_model import LSTMTrainer
from .rl_agent import RLAgent, TradingEnvironment
from .strategy import EnsembleStrategy
from .trader import Trader

LOGGER = logging.getLogger(__name__)


@dataclass
class RecalibrationThresholds:
    """Thresholds to determine when the strategy requires recalibration."""

    min_total_return: float | None = None
    min_win_rate: float | None = None

    def describe(self) -> str:
        parts: list[str] = []
        if self.min_total_return is not None:
            parts.append(f"min_total_return={self.min_total_return:.4f}")
        if self.min_win_rate is not None:
            parts.append(f"min_win_rate={self.min_win_rate:.4f}")
        return ", ".join(parts) if parts else "no thresholds"

    def unmet_conditions(self, result: BacktestResult) -> list[str]:
        issues: list[str] = []
        if self.min_total_return is not None and result.total_return < self.min_total_return:
            issues.append(
                "total_return %.2f%% < threshold %.2f%%"
                % (result.total_return * 100, self.min_total_return * 100)
            )
        if self.min_win_rate is not None and result.win_rate < self.min_win_rate:
            issues.append(
                "win_rate %.2f%% < threshold %.2f%%"
                % (result.win_rate * 100, self.min_win_rate * 100)
            )
        return issues


def _interval_to_minutes(interval: str) -> int:
    """Convert Binance kline interval strings to minutes."""

    interval = interval.strip()
    if not interval:
        raise ValueError("Interval string must not be empty")

    unit_multipliers = {
        "m": 1,
        "h": 60,
        "d": 60 * 24,
        "w": 60 * 24 * 7,
    }

    digits = ""
    unit = ""
    for char in interval:
        if char.isdigit():
            if unit:
                raise ValueError(f"Invalid interval format: {interval}")
            digits += char
        else:
            unit += char

    if not digits or not unit:
        raise ValueError(f"Invalid interval format: {interval}")

    if unit not in unit_multipliers:
        raise ValueError(f"Unsupported interval unit: {unit}")

    return int(digits) * unit_multipliers[unit]


def _load_data(loader: MarketDataLoader, interval: str, samples: int) -> pd.DataFrame:
    interval_minutes = _interval_to_minutes(interval)
    lookback_minutes = samples * interval_minutes
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
    close_prices = close_series.to_numpy(dtype=np.float32)
    future_prices = np.concatenate([close_prices[1:], close_prices[-1:]]).astype(
        np.float32
    )
    lstm_predictions = np.zeros((len(normalized_df), 3))

    for idx in range(CONFIG.lstm_sequence_length, len(normalized_df)):
        window = feature_df.iloc[idx - CONFIG.lstm_sequence_length : idx]
        prediction = trainer.infer(model, window)
        lstm_predictions[idx] = prediction

    env = TradingEnvironment(
        normalized=normalized,
        current_prices=close_prices,
        next_prices=future_prices,
        lstm_predictions=lstm_predictions,
        trade_size=CONFIG.trade_size,
    )
    agent = RLAgent()
    agent.train(env)


def _run_single_backtest(loader: MarketDataLoader) -> BacktestResult:
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

    price_series = frame_5m.set_index("open_time")["close"].loc[common_index]

    try:
        model = trainer.load_model()
        if len(normalized_df) > CONFIG.lstm_sequence_length:
            lstm_predictions = np.zeros((len(normalized_df), 3))
            for idx in range(CONFIG.lstm_sequence_length, len(normalized_df)):
                window = feature_df.iloc[idx - CONFIG.lstm_sequence_length : idx]
                prediction = trainer.infer(model, window)
                lstm_predictions[idx] = prediction
            LOGGER.info(
                "Backtest LSTM predictions aligned for %d samples (last timestamp: %s)",
                len(normalized_df),
                normalized_df.index[-1],
            )
        else:
            lstm_predictions = None
    except FileNotFoundError:
        LOGGER.warning("LSTM model not found, proceeding without predictions")
        lstm_predictions = None

    dqn_actions = None
    try:
        rl_agent = RLAgent()
        dqn_model = rl_agent.load()
        dqn_actions = RLAgent.infer_actions(
            dqn_model,
            normalized_df,
            price_series,
            lstm_predictions,
        )
        LOGGER.info("DQN policy aligned for %d timestamps", len(dqn_actions))
    except FileNotFoundError:
        LOGGER.warning("DQN policy not found, proceeding without reinforcement signals")
    except Exception as exc:
        LOGGER.exception("Failed to generate DQN actions for backtest: %s", exc)

    strategy = EnsembleStrategy()
    trader = Trader(live=False)
    backtester = Backtester(strategy, trader)
    return backtester.run(frame_5m, frame_15m, lstm_predictions, dqn_actions)


def recalibrate_strategy(
    loader: MarketDataLoader,
    attempt: int,
    thresholds: RecalibrationThresholds | None,
) -> None:
    LOGGER.info(
        "Recalibration attempt %d starting (%s)",
        attempt,
        thresholds.describe() if thresholds else "no thresholds provided",
    )
    LOGGER.info("Attempt %d: retraining LSTM model", attempt)
    train_lstm(loader)
    LOGGER.info("Attempt %d: retraining reinforcement learning agent", attempt)
    train_rl(loader)


def run_backtest(
    loader: MarketDataLoader,
    *,
    auto_recalibrate: bool = False,
    thresholds: RecalibrationThresholds | None = None,
    max_attempts: int = 3,
) -> None:
    if auto_recalibrate and thresholds is None:
        LOGGER.warning(
            "Auto recalibration requested but no thresholds provided; running a single backtest."
        )
        auto_recalibrate = False

    attempt = 1
    final_result: BacktestResult | None = None

    while True:
        LOGGER.info("Starting backtest attempt %d", attempt)
        result = _run_single_backtest(loader)
        LOGGER.info(
            "Backtest attempt %d completed | PnL: %.2f | Win rate: %.2f%% | Total return: %.2f%%",
            attempt,
            result.pnl,
            result.win_rate * 100,
            result.total_return * 100,
        )

        unmet = thresholds.unmet_conditions(result) if thresholds else []
        if not auto_recalibrate or not unmet:
            if thresholds:
                if unmet:
                    LOGGER.warning(
                        "Backtest below thresholds but auto recalibration disabled (%s)",
                        "; ".join(unmet),
                    )
                else:
                    LOGGER.info("Backtest thresholds satisfied after attempt %d", attempt)
            final_result = result
            break

        LOGGER.warning(
            "Backtest metrics below thresholds on attempt %d: %s",
            attempt,
            "; ".join(unmet),
        )

        if attempt >= max_attempts:
            LOGGER.warning(
                "Maximum recalibration attempts (%d) reached without meeting thresholds",
                max_attempts,
            )
            final_result = result
            break

        recalibrate_strategy(loader, attempt + 1, thresholds)
        attempt += 1
        LOGGER.info("Re-running backtest after recalibration attempt %d", attempt)

    if final_result is None:
        return

    for trade in final_result.trades[-10:]:
        LOGGER.info(
            "Trade: %s %s @ %.2f | %s",
            trade.timestamp,
            trade.action,
            trade.price,
            trade.reason,
        )


def run_paper_trading(loader: MarketDataLoader) -> None:
    trader = Trader(live=False)
    strategy = EnsembleStrategy()
    trainer = LSTMTrainer()

    try:
        model = trainer.load_model()
    except FileNotFoundError as exc:
        LOGGER.error("Paper trading requires a trained LSTM model: %s", exc)
        return

    try:
        rl_agent = RLAgent()
        dqn_model = rl_agent.load()
        LOGGER.info("Loaded DQN policy for paper trading")
    except FileNotFoundError:
        dqn_model = None
        LOGGER.warning("Paper trading will run without DQN reinforcement due to missing policy")

    dqn_last_action = 0

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

        dqn_action = None
        if dqn_model is not None and len(normalized) >= CONFIG.rl_window:
            window_norm = normalized.tail(CONFIG.rl_window)
            if len(window_norm) == CONFIG.rl_window:
                observation = np.concatenate(
                    [
                        window_norm.to_numpy(dtype=np.float32).flatten(),
                        np.array([dqn_last_action], dtype=np.float32),
                        lstm_prediction.astype(np.float32),
                    ]
                )
                action_value, _ = dqn_model.predict(observation, deterministic=True)
                dqn_action = int(action_value)
                dqn_last_action = dqn_action

        decision = strategy.decide(row, signals, lstm_prediction, dqn_action=dqn_action)
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
    try:
        rl_agent = RLAgent()
        dqn_model = rl_agent.load()
        LOGGER.info("Loaded DQN policy for live trading")
    except FileNotFoundError:
        dqn_model = None
        LOGGER.warning("Live trading will run without DQN reinforcement due to missing policy")

    dqn_last_action = 0
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

        dqn_action = None
        if dqn_model is not None and len(normalized) >= CONFIG.rl_window:
            window_norm = normalized.tail(CONFIG.rl_window)
            if len(window_norm) == CONFIG.rl_window:
                observation = np.concatenate(
                    [
                        window_norm.to_numpy(dtype=np.float32).flatten(),
                        np.array([dqn_last_action], dtype=np.float32),
                        lstm_prediction.astype(np.float32),
                    ]
                )
                action_value, _ = dqn_model.predict(observation, deterministic=True)
                dqn_action = int(action_value)
                dqn_last_action = dqn_action

        decision = strategy.decide(row, signals, lstm_prediction, dqn_action=dqn_action)
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
    parser.add_argument(
        "--auto-recalibrate",
        action="store_true",
        help="Automatically retrain models and rerun the backtest if performance thresholds are not met",
    )
    parser.add_argument(
        "--min-total-return",
        type=float,
        default=None,
        help="Minimum total return (as a decimal, e.g. 0.05 for 5%) required to skip recalibration",
    )
    parser.add_argument(
        "--min-win-rate",
        type=float,
        default=None,
        help="Minimum win rate (0-1 range) required to skip recalibration",
    )

    args = parser.parse_args(argv)
    configure_logging()
    loader = MarketDataLoader()

    thresholds: RecalibrationThresholds | None = None
    if args.min_total_return is not None or args.min_win_rate is not None:
        thresholds = RecalibrationThresholds(
            min_total_return=args.min_total_return,
            min_win_rate=args.min_win_rate,
        )

    try:
        if args.train_lstm:
            train_lstm(loader)
        if args.train_rl:
            train_rl(loader)
        if args.backtest:
            run_backtest(
                loader,
                auto_recalibrate=args.auto_recalibrate,
                thresholds=thresholds,
            )
        if args.paper:
            run_paper_trading(loader)
        if args.live:
            run_live_trading(loader)
        actions_selected = [
            args.train_lstm,
            args.train_rl,
            args.backtest,
            args.paper,
            args.live,
        ]
        if not any(actions_selected):
            parser.print_help()
            return 1
    except Exception as exc:  # pragma: no cover - high-level safeguard
        LOGGER.exception("Application error: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
