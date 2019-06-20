import numpy as np
import pandas as pd


def actual_to_csv(id, actual, name="submission"):
    df = pd.DataFrame(actual)
    df.index = id
    df.columns = ["value"]
    df.to_csv("submission.csv")


def mape(forecast, actual):
    return np.mean(np.abs((np.array(actual) - np.array(forecast)) / np.array(actual)).mean() * 100)