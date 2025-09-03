import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def directional_accuracy(y_true, y_pred):
    return np.mean(np.sign(y_true) == np.sign(y_pred))

def compute_metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "directional": float(directional_accuracy(y_true, y_pred))
    }

def expanding_r2(y_true, y_pred):
    # returns expanding R² over time (same length as y_true)
    mu = np.mean(y_true)  # baseline const model
    rss = np.cumsum((y_true - y_pred)**2)
    tss = np.cumsum((y_true - mu)**2) + 1e-12
    r2 = 1 - rss / tss
    return r2
