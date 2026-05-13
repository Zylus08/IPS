import pandas as pd
import numpy as np

from prophet import Prophet


class ProphetModel:

    def __init__(self, config):

        self.horizon = config["horizon"]

        self.model = Prophet()

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
            "y": series
        })

        self.model.fit(self.df)

    def predict(self, X):

        future = self.model.make_future_dataframe(
            periods=self.horizon
        )

        forecast = self.model.predict(future)

        preds = forecast["yhat"].values[-self.horizon:]

        return np.tile(preds, (len(X), 1))