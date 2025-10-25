"""Tests for the reinforcement learning trading environment."""

import numpy as np

from bot.config import CONFIG
from bot.rl_agent import TradingEnvironment


def _create_environment_with_prices(prices: np.ndarray) -> TradingEnvironment:
    length = prices.shape[0]
    normalized = np.zeros((length, 1), dtype=np.float32)
    lstm_predictions = np.zeros((length, 3), dtype=np.float32)
    next_prices = np.concatenate([prices[1:], prices[-1:]]).astype(np.float32)
    return TradingEnvironment(
        normalized=normalized,
        current_prices=prices.astype(np.float32),
        next_prices=next_prices,
        lstm_predictions=lstm_predictions,
        trade_size=CONFIG.trade_size,
    )


def test_buying_before_price_rise_yields_higher_reward() -> None:
    window = CONFIG.rl_window
    length = window + 3
    prices = np.full(length, 100.0, dtype=np.float32)
    prices[window + 1] = 110.0  # price jump right after the first actionable step
    prices[window + 2] = 120.0

    env_hold = _create_environment_with_prices(prices.copy())
    env_buy = _create_environment_with_prices(prices.copy())

    _, _ = env_hold.reset()
    _, reward_hold, *_ = env_hold.step(0)  # hold

    _, _ = env_buy.reset()
    _, reward_buy, *_ = env_buy.step(1)  # buy

    assert reward_hold == 0.0
    assert reward_buy > reward_hold
