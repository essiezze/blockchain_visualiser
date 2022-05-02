# !/user/bin/env python
"""
The pre-processing module
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

__author__ = "Zhien Zhang"
__email__ = "zhien.zhang@uqconnect.edu.au"

SEQ_LEN = 100
TRAIN_SPLIT = 0.8
LOOK_AHEAD = 1


class Preprocessor:
    def __init__(self, ref_days=SEQ_LEN, train_split=TRAIN_SPLIT, days_look_ahead=LOOK_AHEAD):
        self.ref_days = ref_days                # the number of days in the past used to make prediction
        self.train_split = train_split          # the percentage of training data, e.g. 0.8
        self.days_look_ahead = days_look_ahead  # the number of days to predict

    def to_sequences(self, data):
        d = []

        for i in range(len(data) - self.ref_days - self.days_look_ahead - 1):
            d.append(data[i: i + self.ref_days + self.days_look_ahead])

        return np.array(d)

    def train_test_split(self, data_raw):
        data = self.to_sequences(data_raw)

        num_train = int(self.train_split * data.shape[0])

        X_train = data[:num_train, :-self.days_look_ahead, :]
        y_train = data[:num_train, -self.days_look_ahead:, :]

        X_test = data[num_train:, :-self.days_look_ahead, :]
        y_test = data[num_train:, -self.days_look_ahead:, :]

        return X_train, y_train, X_test, y_test

    def prepare_train_data(self, raw_history_data: pd.DataFrame):
        """
        Prepare the training and testing data for the model

        :param raw_history_data: a dataframe with columns: Date, Open, High, Low, Close, Adj Close, Volume
        :return: (X_train, y_train, X_test, y_test)
        """
        df = raw_history_data.sort_values("Date")

        # normalise the close price
        scaler = MinMaxScaler()
        close_price = df.Close.values.reshape(-1, 1)
        scaled_close = scaler.fit_transform(close_price)

        # remove nan rows
        scaled_close = scaled_close[~np.isnan(scaled_close)]
        scaled_close = scaled_close.reshape(-1, 1)

        # split into train and test sets
        X_train, y_train, X_test, y_test = self.train_test_split(scaled_close)
        return X_train, y_train, X_test, y_test

