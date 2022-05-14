# !/user/bin/env python
"""
The pre-processing module
"""

import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

__author__ = "Zhien Zhang"
__email__ = "zhien.zhang@uqconnect.edu.au"

# prediction model parameters
REF_DAYS = 100
TRAIN_SPLIT = 0.8
LOOK_AHEAD = 1


DATA_FOLDER = "apps/analysis/.data"
DATE_FORMAT = "%Y-%m-%d"


class Preprocessor:
    def __init__(self, raw_data: pd.DataFrame, ref_days=REF_DAYS, train_split=TRAIN_SPLIT, days_look_ahead=LOOK_AHEAD):
        # configuration
        self.ref_days = ref_days                # the number of days in the past used to make prediction
        self.train_split = train_split          # the percentage of training data, e.g. 0.8
        self.days_look_ahead = days_look_ahead  # the number of days to predict
        self.scaler = MinMaxScaler()

        # data
        self.raw_data = raw_data                # unprocessed historical data
        self.normed_close = None                # the normalised close price

    def _to_sequences(self, data):
        d = []

        for i in range(len(data) - self.ref_days - self.days_look_ahead - 1):
            d.append(data[i: i + self.ref_days + self.days_look_ahead])

        return np.array(d)

    def _train_test_split(self, data_raw):
        data = self._to_sequences(data_raw)

        num_train = int(self.train_split * data.shape[0])

        X_train = data[:num_train, :-self.days_look_ahead, :]
        y_train = data[:num_train, -self.days_look_ahead:, :]

        X_test = data[num_train:, :-self.days_look_ahead, :]
        y_test = data[num_train:, -self.days_look_ahead:, :]

        return X_train, y_train, X_test, y_test

    def prepare_train_test_data(self):
        """
        Prepare the training and testing data for the model

        :param raw_history_data: a dataframe with columns: Date, Open, High, Low, Close, Adj Close, Volume
        :return: (X_train, y_train, X_test, y_test)
        """
        df = self.raw_data.sort_values("Date")

        # normalise the close price
        close_price = df["Close"].values.reshape(-1, 1)
        scaled_close = self.scaler.fit_transform(close_price)

        # remove nan rows
        scaled_close = scaled_close[~np.isnan(scaled_close)]
        self.normed_close = scaled_close.reshape(-1, 1)

        # split into train and test sets
        X_train, y_train, X_test, y_test = self._train_test_split(self.normed_close)
        return X_train, y_train, X_test, y_test

    def get_prediction_input(self):
        df = {
            "Date": self.raw_data["Date"][-self.ref_days:],
            "Close": self.normed_close[-self.ref_days:].flatten(),
            "Type": ["Actual"] * self.ref_days
        }
        return pd.DataFrame(df)


def load_from_data_folder_close():
    files = os.listdir(DATA_FOLDER)
    merged = None

    for f in files:
        name = f.split("-")[0]
        df = pd.read_csv(os.path.join(DATA_FOLDER, f))
        df = df[["Date", "Close"]].set_index("Date")

        if merged is None:
            df = df.rename(columns={"Close": name})
            merged = df
        else:
            intersect_dates = merged.index.intersection(df.index)
            merged = merged.loc[intersect_dates]
            merged[name] = df.loc[intersect_dates]["Close"].values

    return merged


def load_from_data_folder_all():
    files = os.listdir(DATA_FOLDER)
    merged = []

    for f in files:
        name = f.split("-")[0]
        df = pd.read_csv(os.path.join(DATA_FOLDER, f))
        df["Coin"] = name
        merged.append(df)

    return pd.concat(merged)


def get_price_data_path(crypto: str):
    return os.path.join(DATA_FOLDER, f"{crypto}-USD.csv")


def read_price_data(crypto: str):
    path = get_price_data_path(crypto)
    return pd.read_csv(path, parse_dates=["Date"])
