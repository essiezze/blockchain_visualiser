# !/user/bin/env python
"""
Train the predict model with historical data and predict the price 15 days ahead
"""

import argparse
from model import Model
from preprocess import *
import matplotlib.pyplot as plt
import seaborn as sns

__author__ = "Zhien Zhang"
__email__ = "zhien.zhang@uqconnect.edu.au"

import os

PREDICTED = "apps/analysis/.predicted"
METRICS_FOLDER = "apps/analysis/.assets"
DAYS_LOOK_AHEAD = 1
REF_DAYS = 90
TRAIN_SPLIT = 1
EPOCHS = 50


def parse_args():
    parser = argparse.ArgumentParser(description='Predict the price of a cryptocurrency')
    parser.add_argument('--crypto', dest='crypto',
                        help=f'The crypto to predict.')

    args = parser.parse_args()
    return args


def get_cryptos_with_available_results():
    files = os.listdir(PREDICTED)
    cryptos = []
    for f in files:
        name = f.split("_")[0]

        if name not in cryptos:
            cryptos.append(name)

    return cryptos


def get_prediction_data_path(crypto: str):
    return os.path.join(PREDICTED, f"{crypto}_predicted.csv")


def get_error_data_path(crypto: str):
    return os.path.join(PREDICTED, f"{crypto}_error.csv")


def get_training_metrics_path(crypto: str):
    return os.path.join(METRICS_FOLDER, f"training_{crypto}.png")


def main(args):
    name = args.crypto

    # preprocess data
    print("Preprocessing...")
    data = read_price_data(name)
    preprocessor = Preprocessor(data, ref_days=REF_DAYS, days_look_ahead=DAYS_LOOK_AHEAD, train_split=TRAIN_SPLIT)
    X_train, y_train, _, _ = preprocessor.prepare_train_test_data()

    # train
    model = Model(REF_DAYS, DAYS_LOOK_AHEAD)
    model.create_model()
    print("Start training...")
    model.train(X_train, y_train, epochs=EPOCHS)

    # plot and save training metrics
    sns.set_theme()
    plt.plot(model.train_stats.history["loss"])
    plt.plot(model.train_stats.history["val_loss"])
    plt.title(f'The Training Loss of the {name} Price Prediction Model')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Training Loss', 'Validation Loss'], loc='upper left')
    metrics_output_path = get_training_metrics_path(name)
    plt.savefig(metrics_output_path)
    print(f"{metrics_output_path} created")

    # get input data for prediction
    print("Predicting")
    input_df = preprocessor.get_prediction_input()
    predicted_normed = model.recursively_predict(input_df)
    predicted_only_normed = predicted_normed[predicted_normed["Type"] == "Predicted"]
    error_band = model.get_error_band(predicted_only_normed, preprocessor.scaler)
    pred_inverse = preprocessor.scaler.inverse_transform(predicted_normed["Close"].values[np.newaxis])
    predicted = predicted_normed
    predicted["Close"] = pred_inverse[0]

    # save results
    predicted.to_csv(get_prediction_data_path(name), index=False)
    error_band.to_csv(get_error_data_path(name), index=False)

    # save model
    Model.save_model(model, name)


# starting point of the execution
if __name__ == "__main__":
    args = parse_args()
    main(args)

