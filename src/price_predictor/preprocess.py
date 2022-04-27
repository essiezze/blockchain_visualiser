# !/user/bin/env python
"""
The pre-processing module
"""

import numpy as np

__author__ = "Zhien Zhang"
__email__ = "zhien.zhang@uqconnect.edu.au"

SEQ_LEN = 100


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


