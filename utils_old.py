import torch
from torch import Tensor
from torch.utils.data import TensorDataset
import numpy as np
import math

from gluonts.dataset import Dataset
from gluonts.dataset.common import ListDataset


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


def create_forecasting_tensors(
    ts: Tensor | np.ndarray, context_length: int, prediction_length: int, n_samples: int
) -> tuple[Tensor, Tensor, Tensor]:
    # ts, mus and vars are assumed to be of the same shape (ts_length, n_features)
    ts = torch.from_numpy(ts)

    starting_indices = np.random.randint(  # the biggest index we can start with is ts_length - ts_piece_length - len_to_pred
        0, ts.shape[0] - context_length - prediction_length, (n_samples,)
    )
    ts_indices = [
        torch.arange(start, start + context_length) for start in starting_indices
    ]
    ts_x = [ts[ts_ind, :] for ts_ind in ts_indices]
    ts_y = [
        ts[start + context_length : start + context_length + prediction_length, :]
        for start in starting_indices
    ]

    indices = torch.stack([Tensor(el) for el in ts_indices])
    xs = torch.stack(ts_x)  # type: ignore we are sure ts_x is a list of Tensors
    ys = torch.stack(ts_y)  # type: ignore we are sure ts_y is a list of Tensors

    return indices, xs, ys


def create_forecasting_dataset(
    ts: Tensor, context_length: int, prediction_length: int, n_samples: int
) -> TensorDataset:
    indices, xs, ys = create_forecasting_tensors(
        ts, context_length, prediction_length, n_samples
    )
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


def get_index_from_Period(start, start_forecast) -> int:
    # start and start_forecast are two tensor of shapes (batch) of pd.Periods
    # we want to get a tensor of starting indices so that we are able to retrieve the correct mu from the normalizer
    return (start_forecast - start).n


def initialize_gluonts_dataset(
    ts: list[np.ndarray],
    means: list[np.ndarray],
    freq: str,
    starts: list,
):
    return ListDataset(
        [  # feat_dynamic_real must be (n_features, ts_length)
            {"target": ts_i, "feat_dynamic_real": m.T, "start": start}
            for ts_i, m, start in zip(ts, means, starts)
        ],
        freq=freq,
    )


def create_dataset_for_mean_layer(
    ts: list[np.ndarray | Tensor],
    mean_ts: list[np.ndarray | Tensor],
    context_el: int,
    prediction_length: int,
    n_el_per_ts: int,
) -> tuple[np.ndarray | Tensor, np.ndarray | Tensor]:
    """
    This function takes as input the original time-series dataset (as a list of time-series)
    and the dataset of the means computed by the GAS normalizer.
    It returns a tuple of two arrays (either numpy or torch.Tensor), containing xs and ys for the mean layer.
    The xs are windows of the means time series of shape (n_el, context_el).
    The ys are windows of the original time series of shape (n_el, prediction_length).
    n_el is len(ts) * n_el_per_ts
    """
    xs_list, ys_list = [], []
    for ts_i, mean_i in zip(ts, mean_ts):
        # we must compute the starting indices for each window
        ts_len = ts_i.shape[0]
        max_start = ts_len - context_el - prediction_length
        indices = np.random.randint(0, max_start, n_el_per_ts)
        for i in indices:
            xs_list.append(mean_i[i : i + context_el])
            ys_list.append(ts_i[i + context_el : i + context_el + prediction_length])

    if isinstance(ts[0], np.ndarray):
        xs = np.array(xs_list)
        ys = np.array(ys_list)
    elif isinstance(ts[0], Tensor):
        xs = torch.stack(xs_list)
        ys = torch.stack(ys_list)
    else:
        raise TypeError(
            "A single time-series must be a list of np.ndarray or torch.Tensor"
        )

    return xs, ys
