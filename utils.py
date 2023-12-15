import pickle
import os
import json
from datetime import datetime
from distutils.util import strtobool

import pandas as pd

from gluonts.dataset.repository import get_dataset as gluonts_get_dataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper

from gluonts.dataset.common import ListDataset, MetaData, TrainDatasets
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.artificial import RecipeDataset, recipe as rcp


def init_folder(folder_name: str) -> str:
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name


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
    """
    # squeeze everything to avoid overflow problems
    squeeze_factor = np.mean(list(dataset.train)[0]["target"][:context_length])
    train_dataset = []
    for el in dataset.train:
        el["target"] = el["target"] / squeeze_factor
        train_dataset.append(el)
    test_dataset = []
    for el in dataset.test:
        el["target"] = el["target"] / squeeze_factor
        test_dataset.append(el)
    """
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
    ) = convert_tsf_to_dataframe(f"{dataset_name}.tsf", "NaN", "series_value")

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
                FieldName.START: pd.Timestamp(train_start_time),  # freq=freq),
            }
        )

        test_series_full_list.append(
            {
                FieldName.TARGET: series_data,
                FieldName.START: pd.Timestamp(train_start_time),  # freq=freq),
            }
        )

    train_ds = ListDataset(train_series_full_list, freq=freq)  # type:ignore not my code
    test_ds = ListDataset(test_series_full_list, freq=freq)  # type:ignore not my code

    return train_ds, test_ds, freq, seasonality
