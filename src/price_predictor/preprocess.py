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


def to_sequences(data, seq_len):
    d = []

    for i in range(len(data) - seq_len):
        d.append(data[i: i + seq_len])

    return np.array(d)


def train_test_split(data_raw, seq_len, train_split):

    data = to_sequences(data_raw, seq_len)

    num_train = int(train_split * data.shape[0])

    X_train = data[:num_train, :-1, :]
    y_train = data[:num_train, -1, :]

    X_test = data[num_train:, :-1, :]
    y_test = data[num_train:, -1, :]

    return X_train, y_train, X_test, y_test


def prepare_train_data(raw_history_data: pd.DataFrame):
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
    X_train, y_train, X_test, y_test = train_test_split(scaled_close, SEQ_LEN, train_split=TRAIN_SPLIT)
    return X_train, y_train, X_test, y_test

