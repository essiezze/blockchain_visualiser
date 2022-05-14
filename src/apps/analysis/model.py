# !/user/bin/env python
"""
The LSTM model to predict the price of cryptocurrencies
"""

from tensorflow import keras

__author__ = "Zhien Zhang"
__email__ = "zhien.zhang@uqconnect.edu.au"

UNITS = 4
DROPOUT = 0.3
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
        model.add(keras.layers.Activation('sigmoid'))

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

    def validate(self, X_test, y_test):
        return self.layers.evaluate(X_test, y_test)

    @staticmethod
    def load_model():
        layers = keras.models.load_model(MODEL_PATH)
        model = Model()
        model.layers = layers
        return model

    def save_model(self):
        self.layers.save(MODEL_PATH)

