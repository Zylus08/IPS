# run_experiment.py
import numpy as np
import pandas as pd
from utils.config import BASE_CONFIG
from utils.data_loader import train_val_test_split, create_windows
from utils.metrics import mae, rmse, mape
import time
# import models
from models.tsmixer import TSMixerModel
from models.patchtst_wrapper import PatchTSTModel
from models.timesnet_wrapper import TimesNetModel
from models.nhits_wrapper import NHITSModel
# from models.prophet_wrapper import ProphetModel
# from models.chronos_wrapper import ChronosModel
# # add others here
import torch

torch.set_float32_matmul_precision('medium')

MODELS = {
    "PatchTST": PatchTSTModel,
    "TimesNet": TimesNetModel,
    # "NHITS": NHITSModel,
    # "Prophet": ProphetModel,
    # "Chronos": ChronosModel
}

DATASETS = {
    "dominick_dataset": r"e:\IPS\dataset_marked\dominick_dataset.tsf",
    "m1_monthly_dataset": r"e:\IPS\dataset_marked\m1_monthly_dataset.tsf",
    "m4_hourly_dataset": r"e:\IPS\dataset_marked\m4_hourly_dataset.tsf",
    "m3_yearly_dataset": r"e:\IPS\dataset_marked\m3_yearly_dataset.tsf",
}

np.random.seed(42)
torch.manual_seed(42)

def load_series(path):
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        return df.iloc[:, 1].values

    elif path.endswith(".tsf"):
        return load_tsf(path)

    else:
        raise ValueError(f"Unsupported file format: {path}")

def load_tsf(path):
    data = []
    with open(path, "r") as f:
        lines = f.readlines()

    start = False
    for line in lines:
        if "@data" in line:
            start = True
            continue

        if start:
            parts = line.strip().split(":")
            if len(parts) < 2:
                continue

            series = parts[-1]
            values = [float(x) for x in series.split(",") if x != "?"]

            if len(values) > 0:
                data.append(values)

    # return first series (or average)
    if len(data) == 0:
        raise ValueError(f"No valid data in {path}")

    # choose longest valid series
    longest_series = max(data, key=len)

    return np.array(longest_series)


def run_pipeline(mode="dataset_first"):
    results = []

    if mode == "dataset_first":
        for dname, path in DATASETS.items():

            series = load_series(path)
            train, val, test = train_val_test_split(series, BASE_CONFIG)
            
            mean = train.mean()
            std = train.std() + 1e-8

            train = (train - mean) / std
            test = (test - mean) / std

            X_train, y_train = create_windows(train, BASE_CONFIG["window_size"], BASE_CONFIG["horizon"])
            X_test, y_test = create_windows(test, BASE_CONFIG["window_size"], BASE_CONFIG["horizon"])

            if len(X_train) == 0 or len(X_test) == 0:
                print(f"Skipping {dname} due to insufficient length")
                continue

            for mname, ModelClass in MODELS.items():
                model = ModelClass(config=BASE_CONFIG)

                try:
                    start = time.time()

                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)

                    end = time.time()

                except Exception as e:
                    print(f"Error in {mname} on {dname}: {e}")
                    continue

                # compare only last forecast window
                y_true_last = y_test[-1]

                # flatten prediction correctly
                if preds.ndim == 2:
                    y_pred_last = preds[0]
                else:
                    y_pred_last = preds

                results.append({
                        "Param": "baseline",
                        "Value": "default",
                        "Dataset": dname,
                        "Model": mname,
                        "MAE": mae(y_true_last, y_pred_last),
                        "RMSE": rmse(y_true_last, y_pred_last),
                        "MAPE": mape(y_true_last, y_pred_last),
                        "Train_Time": end - start})

    elif mode == "model_first":
        for mname, ModelClass in MODELS.items():
            for dname, path in DATASETS.items():

                series = load_series(path)
                train, val, test = train_val_test_split(series, BASE_CONFIG)

                mean = train.mean()
                std = train.std() + 1e-8

                train = (train - mean) / std
                test = (test - mean) / std

                X_train, y_train = create_windows(train, BASE_CONFIG["window_size"], BASE_CONFIG["horizon"])
                X_test, y_test = create_windows(test, BASE_CONFIG["window_size"], BASE_CONFIG["horizon"])

                model = ModelClass(config=BASE_CONFIG)

                try:
                    start = time.time()

                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)

                    end = time.time()

                except Exception as e:
                    print(f"Error in {mname} on {dname}: {e}")
                    continue

                # compare only last forecast window
                y_true_last = y_test[-1]

                # flatten prediction correctly
                if preds.ndim == 2:
                    y_pred_last = preds[0]
                else:
                    y_pred_last = preds

                if len(y_true_last) != len(y_pred_last):
                    min_len = min(
                        len(y_true_last),
                        len(y_pred_last)
                    )

                    y_true_last = y_true_last[:min_len]
                    y_pred_last = y_pred_last[:min_len]

                results.append({
                        "Param": "baseline",
                        "Value": "default",
                        "Dataset": dname,
                        "Model": mname,
                        "MAE": mae(y_true_last, y_pred_last),
                        "RMSE": rmse(y_true_last, y_pred_last),
                        "MAPE": mape(y_true_last, y_pred_last),
                        "Train_Time": end - start
                })

    results_df = pd.DataFrame(results)
    results_df.to_csv("results.csv", index=False)

    print(results_df)

def run_ablation(param_name, values, mode="dataset_first"):
    results = []

    for val in values:
        config = BASE_CONFIG.copy()
        config[param_name] = val

        print(f"\nRunning {param_name} = {val}")

        if mode == "dataset_first":
            for dname, path in DATASETS.items():

                series = load_series(path)
                train, _, test = train_val_test_split(series, config)

                mean = train.mean(axis=0)
                std = train.std(axis=0) + 1e-8

                train = (train - mean) / std
                test = (test - mean) / std

                X_train, y_train = create_windows(
                    train,
                    config["window_size"],
                    config["horizon"]
                )

                X_test, y_test = create_windows(
                    test,
                    config["window_size"],
                    config["horizon"]
                )

                if len(X_train) == 0 or len(X_test) == 0:
                    continue

                for mname, ModelClass in MODELS.items():
                    model = ModelClass(config)

                    try:
                        start = time.time()

                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)

                        end = time.time()

                    except Exception as e:
                        print(f"Error in {mname} on {dname}: {e}")
                        continue

                    # compare only last forecast window
                    y_true_last = y_test[-1]

                    # flatten prediction correctly
                    if preds.ndim == 2:
                        y_pred_last = preds[0]
                    else:
                        y_pred_last = preds

                    if len(y_true_last) != len(y_pred_last):

                        min_len = min(
                            len(y_true_last),
                            len(y_pred_last)
                        )

                        y_true_last = y_true_last[:min_len]
                        y_pred_last = y_pred_last[:min_len]

                    results.append({
                            "Param": param_name,
                            "Value": val,
                            "Dataset": dname,
                            "Model": mname,
                            "MAE": mae(y_true_last, y_pred_last),
                            "RMSE": rmse(y_true_last, y_pred_last),
                            "MAPE": mape(y_true_last, y_pred_last),
                            "Train_Time": end - start
                    })

    return pd.DataFrame(results)


if __name__ == "__main__":

    df_all = []

    df_all.append(run_ablation("window_size", [48, 96, 192]))
    df_all.append(run_ablation("horizon", [12, 24, 48]))
    df_all.append(run_ablation("hidden_dim", [64, 128, 256]))
    df_all.append(run_ablation("layers", [2, 3, 4]))
    df_all.append(run_ablation("lr", [1e-3, 1e-4]))
    df_all.append(run_ablation("dropout", [0.1, 0.3]))

    final_df = pd.concat(df_all, ignore_index=True)
    final_df = final_df.sort_values(by=["Param", "Dataset", "Model"])

    final_df.to_csv("ablation_results.csv", index=False)

    print(final_df)
    best_df = final_df.loc[
        final_df.groupby(
            ["Param", "Dataset", "Model"]
        )["RMSE"].idxmin()
    ]

    best_df.to_csv(
    "best_results.csv",
    index=False
    )