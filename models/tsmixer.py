# models/tsmixer.py

from models.base_model import BaseModel
import numpy as np


class TSMixerModel(BaseModel):

    def __init__(self, config):
        self.config = config

        self.window = config["window_size"]
        self.horizon = config["horizon"]
        self.hidden_dim = config["hidden_dim"]
        self.layers = config["layers"]
        self.lr = config["lr"]
        self.dropout = config["dropout"]

        self.mean = None

    def fit(self, X, y):
        """
        Dummy training step.
        Stores mean target value.
        """

        self.mean = np.mean(y)

    def predict(self, X):
        """
        Dummy prediction.
        Predicts mean value for all horizons.
        """

        return np.ones(
            (len(X), self.horizon)
        ) * self.mean