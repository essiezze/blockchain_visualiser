# !/user/bin/env python
"""
The LSTM model to predict the price of cryptocurrencies
"""

from tensorflow import keras
from preprocess import SEQ_LEN

__author__ = "Zhien Zhang"
__email__ = "zhien.zhang@uqconnect.edu.au"

UNITS = 4
DROPOUT = 0.2


def get_model():
    model = keras.Sequential()
    model.add(keras.layers.LSTM(UNITS, input_shape=(SEQ_LEN - 1, 1)))
    model.add(keras.layers.Dropout(DROPOUT))
    model.add(keras.layers.LSTM(UNITS))
    model.add(keras.layers.Dropout(DROPOUT))
    model.add(keras.layers.LSTM(UNITS))
    model.add(keras.layers.Dropout(DROPOUT))
    model.add(keras.layers.LSTM(UNITS))
    model.add(keras.layers.Dense(units=5))
    model.add(keras.layers.Activation('linear'))

    model.compile(optimizer="adagrad", loss="mean_squared_error")

    return model

