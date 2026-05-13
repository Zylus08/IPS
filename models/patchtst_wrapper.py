import pandas as pd
import numpy as np

from neuralforecast.core import NeuralForecast
from neuralforecast.models import PatchTST

class PatchTSTModel:

    def __init__(self, config):

        self.horizon = config["horizon"]

        self.model = NeuralForecast(
            models=[
                PatchTST(
                    h=self.horizon,
                    input_size=config["window_size"],
                    hidden_size=config["hidden_dim"],
                    n_heads=4,
                    dropout=config["dropout"],
                    learning_rate=config["lr"],
                    max_steps=100
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

        print("\nFORECAST DF:")
        print(forecasts.head())

        print("\nFORECAST SHAPE:")
        print(forecasts.shape)

        # prediction column name
        pred_col = forecasts.columns[-1]

        # extract prediction values
        pred_values = forecasts[pred_col].values

        print("\nPRED VALUES SHAPE:")
        print(pred_values.shape)

        # ensure exact horizon length
        pred_values = pred_values[:self.horizon]

        preds = np.tile(
            pred_values,
            (len(X), 1)
        )

        print("\nFINAL PREDS SHAPE:")
        print(preds.shape)

        return preds