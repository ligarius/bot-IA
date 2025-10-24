"""LSTM model pipeline for trend prediction."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from .config import CONFIG

LOGGER = logging.getLogger(__name__)


@dataclass
class LSTMConfig:
    sequence_length: int = CONFIG.lstm_sequence_length
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-3


class LSTMTrainer:
    """High-level wrapper around the TensorFlow LSTM pipeline."""

    def __init__(self, lstm_config: LSTMConfig | None = None) -> None:
        self.config = lstm_config or LSTMConfig()
        CONFIG.model_dir.mkdir(parents=True, exist_ok=True)

    def _build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=input_shape),
                tf.keras.layers.LSTM(128, return_sequences=True),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.LSTM(64),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(3, activation="softmax"),
            ]
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        return model

    @staticmethod
    def _create_sequences(features: pd.DataFrame, labels: pd.Series, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        label_array = labels.to_numpy()
        feature_array = features.to_numpy()
        for i in range(sequence_length, len(features)):
            X.append(feature_array[i - sequence_length : i])
            y.append(label_array[i])
        return np.array(X), tf.keras.utils.to_categorical(np.array(y), num_classes=3)

    @staticmethod
    def _create_labels(frame: pd.DataFrame) -> pd.Series:
        future_close = frame["close"].shift(-1)
        delta = (future_close - frame["close"]) / frame["close"]
        labels = pd.cut(delta, bins=[-np.inf, -0.001, 0.001, np.inf], labels=[0, 1, 2])
        return labels.cat.codes

    def train(self, features: pd.DataFrame) -> dict:
        labels = self._create_labels(features)
        features = features.dropna()
        labels = labels.loc[features.index]

        valid_mask = labels.isin({0, 1, 2})
        if not valid_mask.all():
            features = features.loc[valid_mask]
            labels = labels.loc[valid_mask]

        unique_labels = set(labels.unique())
        if not unique_labels.issubset({0, 1, 2}):
            raise ValueError(f"Unexpected labels encountered: {unique_labels}")
        LOGGER.debug("Unique labels after filtering: %s", sorted(unique_labels))

        X, y = self._create_sequences(features, labels, self.config.sequence_length)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = self._build_model(input_shape=(self.config.sequence_length, features.shape[1]))
        history = model.fit(
            X_train,
            y_train,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_split=0.1,
            verbose=0,
        )

        y_pred = model.predict(X_test, verbose=0)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_test_labels = np.argmax(y_test, axis=1)

        metrics = {
            "accuracy": accuracy_score(y_test_labels, y_pred_labels),
            "precision": precision_score(y_test_labels, y_pred_labels, average="macro", zero_division=0),
            "recall": recall_score(y_test_labels, y_pred_labels, average="macro", zero_division=0),
            "history": history.history,
        }

        model_path = CONFIG.model_dir / "lstm_classifier.h5"
        model.save(model_path)
        LOGGER.info("Saved LSTM model to %s", model_path)
        return metrics

    def load_model(self) -> tf.keras.Model:
        model_path = CONFIG.model_dir / "lstm_classifier.h5"
        if not model_path.exists():
            raise FileNotFoundError("Trained LSTM model not found. Run with --train-lstm first.")
        return tf.keras.models.load_model(model_path)

    def infer(self, model: tf.keras.Model, features: pd.DataFrame) -> np.ndarray:
        features = features.dropna().tail(self.config.sequence_length)
        if len(features) < self.config.sequence_length:
            raise ValueError("Insufficient data for LSTM inference")
        sequence = features.to_numpy()[-self.config.sequence_length :]
        sequence = np.expand_dims(sequence, axis=0)
        prediction = model.predict(sequence, verbose=0)
        return prediction.squeeze()
