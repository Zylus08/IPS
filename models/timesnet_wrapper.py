import pandas as pd
import numpy as np

from neuralforecast.models import TimesNet
from neuralforecast import NeuralForecast


class TimesNetModel:

    def __init__(self, config):

        self.horizon = config["horizon"]

        self.model = NeuralForecast(
            models=[
                TimesNet(
                    h=self.horizon,
                    input_size=config["window_size"],
                    hidden_size=config["hidden_dim"],
                    dropout=config["dropout"],
                    learning_rate=config["lr"],
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

        # take ONLY prediction column
        pred_col = forecasts.columns[-1]

        pred_values = forecasts[pred_col].astype(float).to_numpy()

        pred_values = pred_values[:self.horizon]

        return pred_values