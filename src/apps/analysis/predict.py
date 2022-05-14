# !/user/bin/env python
"""
Train the predict model with historical data and predict the price 15 days ahead
"""

import argparse

__author__ = "Zhien Zhang"
__email__ = "zhien.zhang@uqconnect.edu.au"

import os

PREDICTED = "apps/analysis/.predicted"


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


def main(args):
    pass


# starting point of the execution
if __name__ == "__main__":
    pass

