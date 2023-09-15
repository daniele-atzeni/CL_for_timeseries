from torch import Tensor
import numpy as np
import math


def generate_timeseries(growing):
    """
    Generate data coming from a dumped sinusoid and return lists
    """

    # Define constants
    if growing:
        A = 1
        b = 0.2
    else:
        A = 130
        b = -0.2

    f = 1
    phi = 0
    w = 2 * math.pi * f
    T = 1 / f
    dt = T / 100
    t = np.arange(0, 25 * T, dt)
    y = A * np.exp(b * t) * np.sin(w * t + phi) + 50

    # Add an increasing gaussian noise to the data
    noise = np.random.normal(0, 2, len(y))
    y = y + noise

    return t, y


def generate_dataset(y, ts_length, el_to_predict):
    """
    Takes a "signal" as a list and returns two torch Tensor containing
    - the inputs with shape (?, ts_length)
    - the labels with shape (?, el_to_predict)
    Question marks are there because it depends on len(y) and ts_length
    """

    ys = []
    preds = []
    for i in range(0, len(y) - ts_length, el_to_predict):
        ys.append(y[i : i + ts_length])
        preds.append(y[i + ts_length : i + ts_length + el_to_predict])

    # converting from a list of np.arrays is slow
    ys = np.array(ys)
    preds = np.array(preds)
    #
    ys = Tensor(ys)  # shape (?, ts_length)
    preds = Tensor(preds)  # shape (?, el_to_predict)
    # we must unsqueeze the last dimension since n_features = 1
    return ys.unsqueeze(-1), preds


def generate_dataset_multivariate(ts, ts_length, el_to_predict):
    """
    Takes a single (multivariate) time series as a tensor (n_features, tot_length)
    and returns two torch Tensor containing
    - the inputs with shape (?, n_features, ts_length)
    - the labels with shape (?, n_features, el_to_predict)
    Question marks are there because it depends on len(y) and ts_length
    """
    _, tot_length = ts.shape

    xs = []
    ys = []

    # we split the ts into pieces of length ts_length + el_to_predict
    combined_ts_length = ts_length + el_to_predict

    for i in range(
        combined_ts_length, tot_length, combined_ts_length
    ):  # we won't consider the final piece of the ts if numbers don't match
        xs.append(
            ts[:, i - combined_ts_length : i - el_to_predict]
        )  # we go "backward" in order to avoid the last tricky piece of the ts
        ys.append(ts[:, i - el_to_predict : i])

    # converting list[np.array] -> Tensor is slow, hence np.array
    xs = Tensor(np.array(xs))
    ys = Tensor(np.array(ys))

    return xs, ys
