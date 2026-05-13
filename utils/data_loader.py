# utils/data_loader.py

import numpy as np

def train_val_test_split(series, config):
    n = len(series)

    train_end = int(n * config["train_ratio"])
    val_end = int(n * (config["train_ratio"] + config["val_ratio"]))

    return (
        series[:train_end],
        series[train_end:val_end],
        series[val_end:]
    )


def create_windows(series, input_window, horizon):
    X, y = [], []

    for i in range(len(series) - input_window - horizon):
        X.append(series[i:i+input_window])
        y.append(series[i+input_window:i+input_window+horizon])

    return np.array(X), np.array(y) 