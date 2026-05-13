import torch
import numpy as np

from chronos import ChronosPipeline


class ChronosModel:

    def __init__(self, config):

        self.horizon = config["horizon"]

        self.pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small",
            device_map="cpu"
        )

    def fit(self, X, y):
        pass

    def predict(self, X):

        preds = []

        for seq in X:

            forecast = self.pipeline.predict(
                torch.tensor(seq),
                prediction_length=self.horizon
            )

            preds.append(
                forecast[0].numpy()
            )

        return np.array(preds)