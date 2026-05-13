import pandas as pd
import numpy as np

from neuralforecast.models import NHITS
from neuralforecast import NeuralForecast


class NHITSModel:

    def __init__(self, config):

        self.horizon = config["horizon"]

        self.model = NeuralForecast(
            models=[
                NHITS(
                    h=self.horizon,
                    input_size=config["window_size"],
                    learning_rate=config["lr"],
                    dropout_prob_theta=config["dropout"],
                    max_steps=15
                )
            ],
            freq='D'
        )

    def fit(self, X, y):

        series = np.concatenate([
            X.flatten(),
            y.flatten()
        ])

        self.df = pd.DataFrame({
            "ds": pd.date_range(
                start="2000-01-01",
                periods=len(series),
                freq="D"
            ),
            "y": series,
            "unique_id": "series1"
        })

        self.model.fit(self.df)

    def predict(self, X):

        forecasts = self.model.predict()

        # model prediction column
        pred_col = forecasts.columns[-1]

        # extract ONLY prediction values
        pred_values = forecasts[pred_col].to_numpy()

        # ensure horizon length
        pred_values = pred_values[:self.horizon]

        return pred_values