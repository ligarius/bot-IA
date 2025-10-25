"""Deep Q-Network agent powered by stable-baselines3."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from .config import CONFIG

LOGGER = logging.getLogger(__name__)


@dataclass
class RLConfig:
    window_size: int = CONFIG.rl_window
    learning_rate: float = 1e-3
    gamma: float = 0.99
    buffer_size: int = 50_000
    learning_starts: int = 1_000
    batch_size: int = 64
    train_freq: int = 4
    target_update_interval: int = 1_000
    exploration_fraction: float = 0.1
    exploration_final_eps: float = 0.02


class TradingEnvironment(gym.Env):
    """Custom Gymnasium environment for reinforcement learning."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        normalized: np.ndarray,
        current_prices: np.ndarray,
        next_prices: np.ndarray,
        lstm_predictions: np.ndarray,
        trade_size: float,
    ) -> None:
        super().__init__()
        self.normalized = normalized
        self.current_prices = current_prices.astype(np.float32)
        self.next_prices = next_prices.astype(np.float32)
        self.lstm_predictions = lstm_predictions
        self.trade_size = trade_size
        self.window_size = CONFIG.rl_window
        self.fee_pct = CONFIG.fee_pct
        self._unnecessary_trade_penalty = self.trade_size * 0.001

        if not (
            len(self.normalized)
            == len(self.current_prices)
            == len(self.next_prices)
            == len(self.lstm_predictions)
        ):
            raise ValueError("Normalized data and price series must have matching lengths")

        extra_dim = 1 + self.lstm_predictions.shape[1]
        obs_dim = self.window_size * normalized.shape[1] + extra_dim
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.reset()

    def _get_observation(self) -> np.ndarray:
        start = self.current_step - self.window_size
        end = self.current_step
        window = self.normalized[start:end]
        flattened = window.flatten().astype(np.float32)
        observation = np.concatenate(
            [
                flattened,
                np.array([self.last_action], dtype=np.float32),
                self.lstm_predictions[end - 1].astype(np.float32),
            ]
        )
        return observation.astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):  # type: ignore[override]
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.last_action = 0
        self.cash = CONFIG.initial_capital
        self.position_size = 0.0
        self.entry_price: float | None = None
        observation = self._get_observation()
        assert (
            observation.shape[0] == self.observation_space.shape[0]
        ), "Observation dimension mismatch"
        return observation, {}

    def step(self, action: int):  # type: ignore[override]
        if self.current_step >= len(self.current_prices) - 1:
            observation = self._get_observation()
            return observation, 0.0, True, False, {}

        current_price = float(self.current_prices[self.current_step])
        next_price = float(self.next_prices[self.current_step])

        portfolio_value_current = self.cash + self.position_size * current_price
        penalty = 0.0

        if action == 1:  # buy
            if self.position_size == 0.0:
                quantity = self.trade_size / current_price
                cost = self.trade_size
                fee = cost * self.fee_pct
                self.cash -= cost + fee
                self.position_size = quantity
                self.entry_price = current_price
            else:
                penalty -= self._unnecessary_trade_penalty
        elif action == 2:  # sell
            if self.position_size > 0.0:
                proceeds = self.position_size * current_price
                fee = proceeds * self.fee_pct
                self.cash += proceeds - fee
                self.position_size = 0.0
                self.entry_price = None
            else:
                penalty -= self._unnecessary_trade_penalty

        portfolio_value_next = self.cash + self.position_size * next_price
        reward = (portfolio_value_next - portfolio_value_current) + penalty

        self.last_action = action
        self.current_step += 1
        done = self.current_step >= len(self.current_prices) - 1

        observation = self._get_observation()
        assert (
            observation.shape[0] == self.observation_space.shape[0]
        ), "Observation dimension mismatch"
        return observation, reward, done, False, {}


class RLAgent:
    """Wraps stable-baselines3 DQN training and inference."""

    def __init__(self, rl_config: RLConfig | None = None) -> None:
        self.config = rl_config or RLConfig()
        CONFIG.model_dir.mkdir(parents=True, exist_ok=True)

    def train(self, env: TradingEnvironment, timesteps: int = 50_000) -> DQN:
        env_vec = DummyVecEnv([lambda: env])
        model = DQN(
            "MlpPolicy",
            env_vec,
            learning_rate=self.config.learning_rate,
            gamma=self.config.gamma,
            buffer_size=self.config.buffer_size,
            learning_starts=self.config.learning_starts,
            batch_size=self.config.batch_size,
            train_freq=self.config.train_freq,
            target_update_interval=self.config.target_update_interval,
            exploration_fraction=self.config.exploration_fraction,
            exploration_final_eps=self.config.exploration_final_eps,
            verbose=0,
        )

        eval_env = DummyVecEnv([lambda: env])
        eval_callback = EvalCallback(eval_env, best_model_save_path=str(CONFIG.model_dir), verbose=0)
        model.learn(total_timesteps=timesteps, callback=eval_callback)
        model.save(CONFIG.model_dir / "dqn_policy")
        LOGGER.info("Saved DQN policy to %s", CONFIG.model_dir / "dqn_policy.zip")
        return model

    def load(self) -> DQN:
        model_path = CONFIG.model_dir / "dqn_policy.zip"
        if not model_path.exists():
            raise FileNotFoundError("Trained DQN policy not found. Run with --train-rl first.")
        return DQN.load(model_path)
