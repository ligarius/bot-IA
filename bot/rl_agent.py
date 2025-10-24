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

    def __init__(self, normalized: np.ndarray, rewards: np.ndarray, lstm_predictions: np.ndarray, trade_size: float) -> None:
        super().__init__()
        self.normalized = normalized
        self.rewards = rewards
        self.lstm_predictions = lstm_predictions
        self.trade_size = trade_size
        self.window_size = CONFIG.rl_window

        obs_dim = self.window_size * normalized.shape[1] + 2
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.reset()

    def _get_observation(self) -> np.ndarray:
        start = self.current_step - self.window_size
        end = self.current_step
        window = self.normalized[start:end]
        flattened = window.flatten()
        observation = np.concatenate(
            [flattened, np.array([self.last_action, self.lstm_predictions[end - 1]])]
        )
        return observation.astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):  # type: ignore[override]
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.last_action = 0
        self.cash = CONFIG.initial_capital
        self.position = 0.0
        observation = self._get_observation()
        return observation, {}

    def step(self, action: int):  # type: ignore[override]
        reward = self.rewards[self.current_step]
        done = self.current_step >= len(self.normalized) - 1

        if action == 1:  # buy
            if self.position == 0:
                self.position = self.trade_size
                self.cash -= self.trade_size
            else:
                reward -= 0.1  # penalize unnecessary buy
        elif action == 2:  # sell
            if self.position > 0:
                self.cash += self.trade_size
                self.position = 0
            else:
                reward -= 0.1

        portfolio_value = self.cash + self.position
        reward += portfolio_value * 0.0001

        self.last_action = action
        self.current_step += 1

        observation = self._get_observation()
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
