import pickle
import os
import json
from datetime import datetime
from distutils.util import strtobool

import numpy as np
import pandas as pd


from gluonts.dataset.repository import get_dataset as gluonts_get_dataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper

from gluonts.dataset.common import ListDataset, MetaData, TrainDatasets
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.artificial import RecipeDataset, recipe as rcp
from torch import Tensor, from_numpy


def init_folder(folder_name: str) -> str:
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
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


def save_list_of_elements(
    folder: str, list_of_els: list, method: str = "pickle"
) -> None:
    for i, el in enumerate(list_of_els):
        if method == "pickle":
            with open(os.path.join(folder, f"ts_{i}.pkl"), "wb") as f:
                pickle.dump(el, f)
        elif method == "json":
            with open(os.path.join(folder, f"ts_{i}.json"), "w") as f:
                json.dump(el, f)
        else:
            raise NotImplementedError(f"Method {method} not implemented.")


def load_list_of_elements(folder: str) -> list:
    res = []
    for i in range(len(os.listdir(folder))):
        with open(os.path.join(folder, f"ts_{i}.pkl"), "rb") as f:
            res.append(pickle.load(f))
    return res


def get_synthetic_dataset(
    dataset_name: str,
    train_length: int,
    prediction_length: int,
    num_timeseries: int,
    recipe: dict | None = None,
    **kwargs,
) -> TrainDatasets:
    if recipe is None:
        daily_smooth_seasonality = rcp.SmoothSeasonality(period=20, phase=0)
        noise = rcp.RandomGaussian(stddev=0.1)
        signal = daily_smooth_seasonality + noise

        recipe = {FieldName.TARGET: signal}

    train_length = 100
    prediction_length = 10
    num_timeseries = 10

    data = RecipeDataset(
        recipe=recipe,
        metadata=MetaData(
            freq="D",
            feat_static_real=[],
            feat_static_cat=[],
            feat_dynamic_real=[],
            prediction_length=prediction_length,
        ),
        max_train_length=train_length,
        prediction_length=prediction_length,
        num_timeseries=num_timeseries,
        # trim_length_fun=lambda x, **kwargs: np.minimum(
        #    int(np.random.geometric(1 / (kwargs["train_length"] / 2))),
        #    kwargs["train_length"],
        # ),
    )

    generated = data.generate()
    assert generated.test is not None
    info = data.dataset_info(generated.train, generated.test)

    return TrainDatasets(info.metadata, generated.train, generated.test)


def get_dataset_and_metadata(
    dataset_name: str, dataset_type: str, dataset_generation_params: dict
) -> tuple:
    """
    This function retrieve a gluonts dataset with its metadata or generate a
    synthetic dataset and its metadata.
    """
    multivariate = dataset_generation_params["multivariate"]

    if dataset_type == "gluonts":
        dataset = gluonts_get_dataset(dataset_name)
    elif dataset_type == "synthetic":
        dataset = get_synthetic_dataset(dataset_name, **dataset_generation_params)
    else:
        raise NotImplementedError(f"Dataset type {dataset_type} not implemented.")

    # we assume no missing values, i.e. time series of equal lengths
    assert len(set([el["target"].shape[0] for el in dataset.train])) == 1, (
        "Time series of different lengths in the train dataset. "
        "This is not supported by the normalizer."
    )
    assert dataset.test is not None, "Test dataset cannot be None"
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
    train_dataset = dataset.train
    test_dataset = dataset.test

    # we use gluonts multivariate grouper to combine the 1D elements of the dataset
    # (list) into a list with a single 2D array (np.ndarray)
    if multivariate:
        n_features = len(dataset.train)
        train_grouper = MultivariateGrouper(max_target_dim=n_features)
        test_grouper = MultivariateGrouper(
            max_target_dim=n_features,
            num_test_dates=len(test_dataset) // n_features,
        )
        train_dataset = train_grouper(dataset.train)
        test_dataset = test_grouper(dataset.test)
    return (
        train_dataset,
        test_dataset,
        prediction_length,
        context_length,
        freq,
        n_features,
    )


# THE FOLLOWING IS FOR LOADING TSF DATASETS AS GLUONTS DATASETS FROM FILES

# tsf data loader

# Converts the contents in a .tsf file into a dataframe and returns it along with other meta-data of the dataset: frequency, horizon, whether the dataset contains missing values and whether the series have equal lengths
#
# Parameters
# full_file_path_and_name - complete .tsf file path
# replace_missing_vals_with - a term to indicate the missing values in series in the returning dataframe
# value_column_name - Any name that is preferred to have as the name of the column containing series values in the returning dataframe

# Example of usage
# loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe("TSForecasting/tsf_data/sample.tsf")

# print(loaded_data)
# print(frequency)
# print(forecast_horizon)
# print(contain_missing_values)
# print(contain_equal_length)


def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(  # type:ignore   not my code
                            pd.Series(numeric_series).array
                        )

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series  # type:ignore   not my code
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )


def get_dataset_from_file(dataset_name, external_forecast_horizon, context_length):
    (
        df,
        frequency,
        forecast_horizon,
        contain_missing_values,
        contain_equal_length,
    ) = convert_tsf_to_dataframe(f"tsf_data/{dataset_name}.tsf", "NaN", "series_value")

    VALUE_COL_NAME = "series_value"
    TIME_COL_NAME = "start_timestamp"
    SEASONALITY_MAP = {
        "minutely": [1440, 10080, 525960],
        "10_minutes": [144, 1008, 52596],
        "half_hourly": [48, 336, 17532],
        "hourly": [24, 168, 8766],
        "daily": 7,
        "weekly": 365.25 / 7,
        "monthly": 12,
        "quarterly": 4,
        "yearly": 1,
    }
    FREQUENCY_MAP = {
        "minutely": "1min",
        "10_minutes": "10min",
        "half_hourly": "30min",
        "hourly": "1H",
        "daily": "1D",
        "weekly": "1W",
        "monthly": "1M",
        "quarterly": "1Q",
        "yearly": "1Y",
    }

    train_series_list = []
    test_series_list = []
    train_series_full_list = []
    test_series_full_list = []
    final_forecasts = []

    if frequency is not None:
        freq = FREQUENCY_MAP[frequency]
        seasonality = SEASONALITY_MAP[frequency]
    else:
        freq = "1Y"
        seasonality = 1

    if isinstance(seasonality, list):
        seasonality = min(seasonality)  # Use to calculate MASE

    # If the forecast horizon is not given within the .tsf file, then it should be provided as a function input
    if forecast_horizon is None:
        if external_forecast_horizon is None:
            raise Exception("Please provide the required forecast horizon")
        else:
            forecast_horizon = external_forecast_horizon

    start_exec_time = datetime.now()

    for index, row in df.iterrows():
        if TIME_COL_NAME in df.columns:
            train_start_time = row[TIME_COL_NAME]
        else:
            train_start_time = datetime.strptime(
                "1900-01-01 00-00-00", "%Y-%m-%d %H-%M-%S"
            )  # Adding a dummy timestamp, if the timestamps are not available in the dataset or consider_time is False

        series_data = row[VALUE_COL_NAME]

        # Creating training and test series. Test series will be only used during evaluation
        train_series_data = series_data[: len(series_data) - forecast_horizon]
        test_series_data = series_data[
            (len(series_data) - forecast_horizon) : len(series_data)
        ]

        train_series_list.append(train_series_data)
        test_series_list.append(test_series_data)

        # We use full length training series to train the model as we do not tune hyperparameters
        train_series_full_list.append(
            {
                FieldName.TARGET: train_series_data,
                FieldName.START: pd.Timestamp(train_start_time, freq=freq),
            }
        )

        test_series_full_list.append(
            {
                FieldName.TARGET: series_data,
                FieldName.START: pd.Timestamp(train_start_time, freq=freq),
            }
        )

    train_ds = ListDataset(train_series_full_list, freq=freq)  # type:ignore not my code
    test_ds = ListDataset(test_series_full_list, freq=freq)  # type:ignore not my code

    return train_ds, test_ds, freq, seasonality
