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


def plot_corr_two(price_1, name_1, price_2, name_2):
    joint = pd.merge(price_1, price_2, on="Date", suffixes=(f"_{name_1}", f"_{name_2}"))
    sns.set_theme()
    sns.scatterplot(data=joint, x=f"Close_{name_1}", y=f"Close_{name_2}")


def load_from_data_folder():
    """
    Load all data files from the data folder

    :return: a dataframe with all cryptocurrency price
    """
    files = os.listdir(DATA)
    result = None

    for f in files:
        name = f.split("-")[0]
        df = pd.read_csv(os.path.join(DATA, f))
        df["Name"] = name
        dfs.append(df)

    return pd.concat(dfs)


