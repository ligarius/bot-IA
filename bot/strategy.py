"""Trading strategy combining indicators and model outputs."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class StrategyDecision:
    action: int
    reason: str


class EnsembleStrategy:
    """Combine indicator signals, LSTM predictions and risk filters."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def decide(
        self,
        indicators: pd.Series,
        signals: pd.Series,
        lstm_prediction: np.ndarray | None = None,
        dqn_action: int | None = None,
    ) -> StrategyDecision:
        score = 0
        reasons = []

        higher_tf_trend = signals.get("higher_tf_trend", 0)
        proba_buy: float | None = None
        proba_sell: float | None = None

        if signals.get("adx_signal", 0) <= 0:
            return StrategyDecision(action=0, reason="Weak trend (ADX filter)")

        if signals.get("macd_hist_signal", 0) > 0:
            score += 1
            reasons.append("MACD histogram bullish")
        elif signals.get("macd_hist_signal", 0) < 0:
            score -= 1
            reasons.append("MACD histogram bearish")

        if signals.get("supertrend_signal", 0) > 0:
            score += 1
            reasons.append("SuperTrend bullish")
        else:
            score -= 1
            reasons.append("SuperTrend bearish")

        if signals.get("ema_cross_signal", 0) > 0:
            score += 1
            reasons.append("EMA cross bullish")
        else:
            score -= 1
            reasons.append("EMA cross bearish")

        if higher_tf_trend > 0:
            score += 1
            reasons.append("Higher TF trend bullish (+1 weight)")
        else:
            score -= 1.5
            reasons.append("Higher TF trend bearish (strong penalty)")

        if signals.get("bollinger_breakout", 0) > 0:
            score += 0.5
            reasons.append("Bollinger breakout up")
        elif signals.get("bollinger_breakout", 0) < 0:
            score -= 0.5
            reasons.append("Bollinger breakout down")

        if lstm_prediction is not None:
            proba_buy = float(lstm_prediction[2])
            proba_sell = float(lstm_prediction[0])
            if proba_buy - proba_sell > self.threshold:
                score += 1
                reasons.append("LSTM predicts upward move")
            elif proba_sell - proba_buy > self.threshold:
                score -= 1
                reasons.append("LSTM predicts downward move")

        if dqn_action is not None:
            if dqn_action == 1:
                score += 1.5
                reasons.append("DQN policy favors BUY")
            elif dqn_action == 2:
                score -= 1.5
                reasons.append("DQN policy favors SELL")
            else:
                reasons.append("DQN policy neutral -> HOLD bias")

        forced_hold = False
        if higher_tf_trend <= 0 and (proba_buy is None or proba_buy < 0.6):
            forced_hold = True
            reasons.append(
                "Higher TF bearish and LSTM buy confidence below 0.60 -> HOLD"
            )

        if forced_hold:
            action = 0
            reasons.append("Final decision: HOLD (trend guard)")
        elif score > 1.5:
            action = 1
            reasons.append("Final decision: BUY")
        elif score < -1.5:
            action = 2
            reasons.append("Final decision: SELL")
        else:
            action = 0
            reasons.append("Final decision: HOLD")

        return StrategyDecision(action=action, reason="; ".join(reasons))
