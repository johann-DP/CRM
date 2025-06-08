"""Train LSTM models for aggregated revenue forecasting."""

from __future__ import annotations

from typing import Tuple

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Reduce TensorFlow verbosity during model creation
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def create_lstm_sequences(series: pd.Series, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return sequences of fixed length for LSTM training.

    Parameters
    ----------
    series:
        Time series of revenue values.
    window_size:
        Number of past observations used to predict the next one.
    """
    values = series.values.reshape(-1, 1)
    X, y = [], []
    for i in range(len(values) - window_size):
        X.append(values[i : i + window_size])
        y.append(values[i + window_size])
    return np.array(X), np.array(y).reshape(-1)


def scale_lstm_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """Scale ``X`` and ``y`` between 0 and 1 using ``MinMaxScaler``."""
    scaler = MinMaxScaler()
    # Fit on both predictors and targets to keep consistency
    scaler.fit(np.vstack([X.reshape(-1, 1), y.reshape(-1, 1)]))
    X_scaled = scaler.transform(X.reshape(-1, 1)).reshape(X.shape)
    y_scaled = scaler.transform(y.reshape(-1, 1)).reshape(-1)
    return X_scaled, y_scaled, scaler


# ---------------------------------------------------------------------------
# Model creation and training
# ---------------------------------------------------------------------------

def build_lstm_model(window_size: int):
    """Return a simple LSTM model with 50 units followed by a Dense output."""
    from keras.models import Sequential
    from keras.layers import Input, LSTM, Dense

    model = Sequential([
        Input(shape=(window_size, 1)),
        LSTM(50),
        Dense(1),
    ])
    model.compile(loss="mse", optimizer="adam")
    return model


def train_lstm_model(
    series: pd.Series,
    *,
    window_size: int,
    epochs: int = 50,
    batch_size: int = 16,
    validation_split: float = 0.1,
):
    """Prepare data, train an LSTM model and return it with its scaler."""
    X, y = create_lstm_sequences(series, window_size)
    X_scaled, y_scaled, scaler = scale_lstm_data(X, y)

    model = build_lstm_model(window_size)
    history = model.fit(
        X_scaled,
        y_scaled,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1,
    )
    return model, scaler, history

