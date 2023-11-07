import pickle
import numpy as np
import os
import json

from gluonts.dataset.repository import get_dataset as gluonts_get_dataset
from torch import Tensor, from_numpy


def init_folder(folder_name: str) -> str:
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    return folder_name


def compute_starting_indices(
    ts_dataset: list[np.ndarray],
    context_length: int,
    prediction_length: int,
    n_samples_per_ts: int,
) -> list[np.ndarray]:
    """
    This function computes n_sample_per_ts starting indices for each time series
    in a dataset.
    Starting indices must be lower than ts_len - context_length - prediction_length.
    """
    indices = []
    for ts in ts_dataset:
        indices.append(
            np.random.randint(
                low=0,
                high=ts.shape[0] - context_length - prediction_length,
                size=n_samples_per_ts,
            )
        )
    return indices


def prepare_dataset_for_mean_layer(
    means: list[np.ndarray],
    dataset: list[np.ndarray],
    context_length: int,
    prediction_length: int,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    This function
    - splits means dataset (list of time series means) and uses them as x for
      the mean layer
    - splits the original dataset (list of time series) and uses them as y for
      the mean layer
    It does so by creating n_test_sample_per_ts starting indices for each ts for
    each start index then takes mean_ts[start: start + context_length] as x and
    ts[start + context_length: start + context_length + prediction_length] as y.
    Inputs are assumed to be lists of 2D arrays (len, n_feat).
    """
    len_dataset = len(dataset)
    assert len_dataset == len(means), "Dataset and means must have the same length"
    n_samples_per_ts = n_samples // len_dataset
    n_features = dataset[0].shape[1]

    # inputs and outputs of the mean layer are 1D
    mean_layer_x = np.empty((n_samples, context_length * n_features))
    mean_layer_y = np.empty((n_samples, prediction_length * n_features))

    # computing starting indices
    start_indices = compute_starting_indices(
        dataset, context_length, prediction_length, n_samples_per_ts
    )

    # slice and fill the arrays
    for i, (ts, mean_ts, start_idxs) in enumerate(zip(dataset, means, start_indices)):
        for j, start_idx in enumerate(start_idxs):
            mean_window_x = mean_ts[start_idx : start_idx + context_length]
            mean_layer_x[i * n_samples_per_ts + j] = mean_window_x.reshape(
                context_length * n_features
            )
            mean_window_y = ts[
                start_idx
                + context_length : start_idx
                + context_length
                + prediction_length
            ]
            mean_layer_y[i * n_samples_per_ts + j] = mean_window_y.reshape(
                prediction_length * n_features
            )

    return mean_layer_x, mean_layer_y


def prepare_dataset_for_torch_model(
    dataset: list[np.ndarray],
    means: list[np.ndarray],
    context_length: int,
    prediction_length: int,
    n_samples: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    This function splits the dataset and the means (both list of time series).
    In particular, it create the starting indices for each time series, then cuses
    them to create inputs for the model (ts_windows and mean_windows)
    and outputs (ts_windows). We always assume ts and means as 2D arrays (len, n_feat).
    """
    len_dataset = len(dataset)
    assert len_dataset == len(means), "Dataset and means must have the same length"
    n_samples_per_ts = n_samples // len_dataset
    n_features = dataset[0].shape[1]

    # computing starting indices
    start_indices = compute_starting_indices(
        dataset, context_length, prediction_length, n_samples_per_ts
    )

    # slice and fill the arrays
    ts_dataset_x = np.empty((n_samples, context_length, n_features))
    mean_dataset_x = np.empty(
        (n_samples, context_length * n_features)
    )  # linear layer inputs are 1D
    ts_dataset_y = np.empty((n_samples, prediction_length, n_features))

    for i, (ts, mean, start_idxs) in enumerate(zip(dataset, means, start_indices)):
        for j, start in enumerate(start_idxs):
            ts_window_x = ts[start : start + context_length]
            ts_dataset_x[i * n_samples_per_ts + j] = ts_window_x  # 2D
            mean_window_x = mean[start : start + context_length]
            mean_dataset_x[i * n_samples_per_ts + j] = mean_window_x.reshape(
                context_length * n_features
            )  # linear layer inputs are 1D
            ts_window_y = ts[
                start + context_length : start + context_length + prediction_length
            ]
            ts_dataset_y[i * n_samples_per_ts + j] = ts_window_y  # 2D

    ts_dataset_x = from_numpy(ts_dataset_x).float()
    mean_dataset_x = from_numpy(mean_dataset_x).float()
    ts_dataset_y = from_numpy(ts_dataset_y).float()
    return ts_dataset_x, mean_dataset_x, ts_dataset_y


def save_list_of_elements(folder: str, list_of_els: list) -> None:
    for i, el in enumerate(list_of_els):
        with open(os.path.join(folder, f"ts_{i}.pkl"), "wb") as f:
            pickle.dump(el, f)


def load_list_of_elements(folder: str) -> list:
    res = []
    for i in range(len(os.listdir(folder))):
        with open(os.path.join(folder, f"ts_{i}.pkl"), "rb") as f:
            res.append(pickle.load(f))
    return res


def get_dataset_and_metadata(dataset_name: str, dataset_type: str):
    """
    This function retrieve a gluonts dataset with its metadata or generate a
    synthetic dataset and its metadata.
    """
    if dataset_type == "gluonts":
        dataset = gluonts_get_dataset(dataset_name)
        # we assume no missing values, i.e. time series of equal lengths
        assert len(set([el["target"].shape[0] for el in dataset.train])) == 1, (
            "Time series of different lengths in the train dataset. "
            "This is not supported by the normalizer."
        )
        # dataset parameters
        assert (
            dataset.metadata.prediction_length is not None
        ), "Prediction length cannot be None"
        prediction_length = dataset.metadata.prediction_length
        context_length = 2 * prediction_length
        assert dataset.metadata.freq is not None, "Frequency length cannot be None"
        freq = dataset.metadata.freq
        n_features = (
            1
            if len(list(dataset.train)[0]["target"].shape) == 1
            else list(dataset.train)[0]["target"].shape[1]
        )
        return dataset, prediction_length, context_length, freq, n_features
    else:
        raise NotImplementedError(
            f"Dataset type {dataset_type} not implemented for dataset {dataset_name}"
        )
