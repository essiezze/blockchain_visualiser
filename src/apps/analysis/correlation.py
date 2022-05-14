# !/user/bin/env python
"""
Calculate price correlation
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

__author__ = "Zhien Zhang"
__email__ = "zhien.zhang@uqconnect.edu.au"

DATA = "./.data"


def from_merged_get_scatter_data(merged, x_name: str):
    """
    Get the data for scatter plot showing the correlation between the price of different cryptocurrency

    :param merged: output dataframe from preprocess.load_from_data_folder_close()
    :param x_name: the name of the crypto in x axis
    :return: a dataframe of data for visualising the correlation
    """
    scatter_data = {
        "Date": [],
        x_name: [],
        "Other": [],
        "Name": []
    }
    num_of_days = len(merged)

    for col in merged.columns:
        if col != x_name:
            scatter_data["Date"].extend(merged.index.to_list())
            scatter_data[x_name].extend(merged[x_name].values)
            scatter_data["Other"].extend(merged[col].values)
            scatter_data["Name"].extend([col] * num_of_days)

    return pd.DataFrame(scatter_data)


def get_corr(raw):
    """
    Get the correlation matrix

    :param raw: output dataframe from preprocess.load_from_data_folder_close()
    :return: the correlation matrix
    """

    return raw.corr()


