# !/user/bin/env python
"""
The LSTM model to predict the price of cryptocurrencies
"""

from preprocess import DATE_FORMAT
import os.path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from tensorflow import keras
from pickle import dump, load

__author__ = "Zhien Zhang"
__email__ = "zhien.zhang@uqconnect.edu.au"

UNITS = 4
DROPOUT = 0.1
BATCH_SIZE = 64
MODEL_PATH = "./model"


class Model:
    def __init__(self, ref_days: int, days_look_ahead: int):
        self.layers = None
        self.ref_days = ref_days
        self.days_look_ahead = days_look_ahead
        self.train_stats = None

    def create_model(self):
        model = keras.Sequential()
        model.add(keras.layers.LSTM(UNITS, input_shape=(self.ref_days, 1), return_sequences=True))
        model.add(keras.layers.Dropout(DROPOUT))
        model.add(keras.layers.LSTM(UNITS, return_sequences=True))
        model.add(keras.layers.Dropout(DROPOUT))
        model.add(keras.layers.LSTM(UNITS))
        model.add(keras.layers.Dense(units=self.days_look_ahead))
        model.add(keras.layers.Activation('linear'))

        optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss="mean_squared_error")

        self.layers = model

    def train(self, X_train, y_train,
              batch_size=BATCH_SIZE,
              validation_split=0.1,
              epochs=25):
        history = self.layers.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=False,
            validation_split=validation_split
        )

        self.train_stats = history

    def evaluate(self, X_test, y_test):
        return self.layers.evaluate(X_test, y_test)

    def predict(self, X):
        return self.layers.predict(X)

    def recursively_predict(self, input, times=15):
        """
        Recursively predict the price of the crypto

        :param input: outputted by preprocessor.get_prediction_input()
        :param times: number of days to predict
        :return: predicted and input data
        """
        close = input["Close"].values[np.newaxis]
        predicted = {
            "Date": [],
            "Close": [],
            "Type": []
        }

        predicted = self._recursively_predict(close, predicted, input["Date"].max(), times)
        return pd.concat([input, pd.DataFrame(predicted)], ignore_index=True)

    def _recursively_predict(self, input, output, last_date, times):
        if times != 0:
            predicted = self.predict(input)[0][0]
            new_date = last_date + timedelta(days=1)
            new_input = np.append(input, predicted)[np.newaxis]
            new_input = new_input[:, -self.ref_days:]

            output["Date"].append(new_date)
            output["Close"].append(predicted)
            output["Type"].append("Predicted")

            return self._recursively_predict(new_input, output, new_date, times - 1)
        else:
            return output

    def get_error_band(self, predicted_data: pd.DataFrame, scaler):
        val_error = self.train_stats.history["val_loss"][-1]
        val_error_geom = []
        for i in range(len(predicted_data)):
            if not val_error_geom:
                val_error_geom.append(val_error)
            else:
                last_val_error = val_error_geom[-1]
                val_error_geom.append(last_val_error * 1.05)

        lower = predicted_data["Close"] - val_error_geom
        upper = predicted_data["Close"] + val_error_geom

        error_band_raw = {
            "Date": predicted_data["Date"],
            "Lower": scaler.inverse_transform(lower.values[np.newaxis])[0],
            "Upper": scaler.inverse_transform(upper.values[np.newaxis])[0],
        }

        return pd.DataFrame(error_band_raw)

    @staticmethod
    def load_model(name):
        path = os.path.join(MODEL_PATH, name)
        with open(path, "rb") as fp:
            model = load(fp)

        return model

    @staticmethod
    def save_model(model, name):
        output_path = os.path.join(MODEL_PATH, name)
        with open(output_path, "wb") as fp:
            dump(model, fp)

