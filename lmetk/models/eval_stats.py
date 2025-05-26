import numpy as np

EPSILON = 1e-7


def r_squared(actual: np.ndarray, predicted: np.ndarray, agg="mean"):
    rss = (_error(actual, predicted) ** 2).sum(1)
    tss = (_error(actual, actual.mean(1, keepdims=True)) ** 2).sum(1)
    r2 = 1.0 - rss / (tss + EPSILON)
    return r2.mean() if agg == "mean" else r2


def _error(actual: np.ndarray, predicted: np.ndarray):
    """Simple error"""
    return actual - predicted


def _percentage_error(actual: np.ndarray, predicted: np.ndarray):
    """
    Percentage error

    Note: result is NOT multiplied by 100
    """
    return _error(actual, predicted) / (actual + EPSILON)


def mse(actual: np.ndarray, predicted: np.ndarray, agg="mean"):
    """Mean Squared Error"""
    e = np.square(_error(actual, predicted))
    return np.mean(e) if agg == "mean" else e


def mae(actual: np.ndarray, predicted: np.ndarray, agg="mean"):
    """Mean Absolute Error"""
    e = np.abs(_error(actual, predicted))
    return np.mean(e) if agg == "mean" else e


def mape(actual: np.ndarray, predicted: np.ndarray, agg="mean"):
    """Mean Absolute Percentage Error"""
    e = np.abs(_percentage_error(actual, predicted))
    return np.mean(e) if agg == "mean" else e


def smape(actual: np.ndarray, predicted: np.ndarray, agg="mean"):
    """Symmetric Mean Absolute Percentage Error"""
    e = (
        2.0
        * np.abs(actual - predicted)
        / ((np.abs(actual) + np.abs(predicted)) + EPSILON)
    )
    return np.mean(e) if agg == "mean" else e
