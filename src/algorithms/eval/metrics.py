"""Metrics for comparing polarization estimates."""

from collections.abc import Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr


def spearman(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Return Spearman rank correlation, or NaN for undefined input."""
    return float(spearmanr(y_true, y_pred).statistic)


def pearson(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Return Pearson correlation, or NaN for undefined input."""
    return float(pearsonr(y_true, y_pred).statistic)


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Return mean absolute error between two equally sized sequences."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.mean(np.abs(true - pred)))


def direction_of_change_agreement(
    y_true: Sequence[float], y_pred: Sequence[float]
) -> float:
    """Return the fraction of adjacent changes with matching direction."""
    true = np.diff(np.asarray(y_true, dtype=float))
    pred = np.diff(np.asarray(y_pred, dtype=float))
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if true.size == 0:
        return float("nan")
    return float(np.mean(np.sign(true) == np.sign(pred)))
