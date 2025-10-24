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
    ) -> StrategyDecision:
        score = 0
        reasons = []

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

        if signals.get("higher_tf_trend", 0) > 0:
            score += 0.5
            reasons.append("Higher TF trend bullish")
        else:
            score -= 0.5
            reasons.append("Higher TF trend bearish")

        if signals.get("bollinger_breakout", 0) > 0:
            score += 0.5
            reasons.append("Bollinger breakout up")
        elif signals.get("bollinger_breakout", 0) < 0:
            score -= 0.5
            reasons.append("Bollinger breakout down")

        if lstm_prediction is not None:
            proba_buy = lstm_prediction[2]
            proba_sell = lstm_prediction[0]
            if proba_buy - proba_sell > self.threshold:
                score += 1
                reasons.append("LSTM predicts upward move")
            elif proba_sell - proba_buy > self.threshold:
                score -= 1
                reasons.append("LSTM predicts downward move")

        if score > 1.5:
            action = 1
            reasons.append("Final decision: BUY")
        elif score < -1.5:
            action = 2
            reasons.append("Final decision: SELL")
        else:
            action = 0
            reasons.append("Final decision: HOLD")

        return StrategyDecision(action=action, reason="; ".join(reasons))
