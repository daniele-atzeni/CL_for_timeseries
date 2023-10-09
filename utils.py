import torch
from torch import Tensor
from torch.utils.data import TensorDataset
import numpy as np
import math

from gluonts.dataset import Dataset


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


"""
def create_forecasting_dataset_old(
    ts: Tensor, mus: Tensor, vars: Tensor, ts_len: int, len_to_pred: int, n_samples: int
) -> TensorDataset:
    # ts, mus and vars are assumed to be of the same shape (ts_length, n_features)
    starting_indices = torch.randint(0, ts.shape[0] - len_to_pred, (n_samples,))
    ts_x = [ts[start : start + ts_len, :] for start in starting_indices]
    mus_x = [mus[start : start + ts_len, :] for start in starting_indices]
    vars_x = [vars[start : start + ts_len, :] for start in starting_indices]
    ts_y = [
        ts[start + ts_len : start + ts_len + len_to_pred, :]
        for start in starting_indices
    ]
    # mus_y = [mus[start+ts_len: start+ts_len+len_to_pred, :] for start in starting_indices]
    # vars_y = [vars[start+ts_len: start+ts_len+len_to_pred, :] for start in starting_indices]
    return TensorDataset(ts_x, mus_x, vars_x, ts_y)  # , mus_y, vars_y)
"""


def create_forecasting_dataset(
    ts: Tensor, ts_len: int, len_to_pred: int, n_samples: int
) -> TensorDataset:
    # ts, mus and vars are assumed to be of the same shape (ts_length, n_features)
    print(
        f"Creating the dataset with {n_samples} samples of shape {ts_len} as input and {len_to_pred} as output"
    )
    print(f"Original ts shape: {ts.shape}")
    starting_indices = np.random.randint(  # the biggest index we can start with is ts_length - ts_piece_length - len_to_pred
        0, ts.shape[0] - ts_len - len_to_pred, (n_samples,)
    )
    ts_indices = [torch.arange(start, start + ts_len) for start in starting_indices]
    print(f"Generated {len(ts_indices)} indices")
    ts_x = [ts[ts_ind, :] for ts_ind in ts_indices]
    print(f"Initialized input with {len(ts_x)} elements of shape {ts_x[0].shape}")
    ts_y = [
        ts[start + ts_len : start + ts_len + len_to_pred, :]
        for start in starting_indices
    ]
    print(f"Initialized output with {len(ts_y)} elements of shape {ts_y[0].shape}")

    indices = torch.stack(ts_indices)
    xs = torch.stack(ts_x)
    ys = torch.stack(ts_y)

    print(f"Stacked tensors")

    return TensorDataset(indices, xs, ys)


def convert_gluon_dataset_to_train_tensor(ts: Dataset, pad: float = 0) -> Tensor:
    list_to_stack = [torch.from_numpy(ts_entry["target"]) for ts_entry in ts]
    # padding if el of different shapes
    max_len = max([ts_entry.shape[0] for ts_entry in list_to_stack])
    for i, el in enumerate(list_to_stack):
        if el.shape[0] < max_len:
            n_el_to_add = max_len - el.shape[0]
            el_to_add = torch.zeros(n_el_to_add) + pad
            list_to_stack[i] = torch.cat([el, el_to_add])
    return torch.stack(list_to_stack)
