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
MODEL_PATH = "./model"


class Model:
    def __init__(self):
        self.layers = None

    def create_model(self):
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

        self.layers = model

    @staticmethod
    def load_model():
        layers = keras.models.load_model(MODEL_PATH)
        model = Model()
        model.layers = layers
        return model

    def save_model(self):
        self.layers.save(MODEL_PATH)

