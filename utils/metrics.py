# utils/metrics.py

import numpy as np

def mape(y_true, y_pred):
    epsilon = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100

def smape(y_true, y_pred):

    epsilon = 1e-8

    numerator = np.abs(y_true - y_pred)

    denominator = (
        np.abs(y_true)
        + np.abs(y_pred)
        + epsilon
    ) / 2

    return np.mean(numerator / denominator) * 100

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))