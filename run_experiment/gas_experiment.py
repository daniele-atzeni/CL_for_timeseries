import os

import numpy as np

from data_manager import GluonTSDataManager

from utils import init_folder

from run_experiment.normalizer_experiment import experiment_normalizer
from run_experiment.mean_layer_experiment import experiment_mean_layer_linear
from run_experiment.dl_model_experiment import experiment_gluonts, experiment_torch


def init_folders_for_normalizer(root_folder: str) -> dict:
    normalizer_folder = init_folder(os.path.join(root_folder, "normalizer"))
    norm_parameters_folder = init_folder(
        os.path.join(normalizer_folder, "normalizer_params")
    )
    norm_ts_folder = init_folder(os.path.join(normalizer_folder, "normalized_ts"))
    means_folder = init_folder(os.path.join(normalizer_folder, "means"))
    vars_folder = init_folder(os.path.join(normalizer_folder, "vars"))
    # train folders
    train_params_folder = init_folder(os.path.join(norm_parameters_folder, "train"))
    train_normalized_folder = init_folder(os.path.join(norm_ts_folder, "train"))
    train_means_folder = init_folder(os.path.join(means_folder, "train"))
    train_vars_folder = init_folder(os.path.join(vars_folder, "train"))
    # test filenames (we will append the index of the time series)
    test_params_folder = init_folder(os.path.join(norm_parameters_folder, "test"))
    test_normalized_folder = init_folder(os.path.join(norm_ts_folder, "test"))
    test_means_folder = init_folder(os.path.join(means_folder, "test"))
    test_vars_folder = init_folder(os.path.join(vars_folder, "test"))

    return {
        "normalizer": normalizer_folder,
        "train_params": train_params_folder,
        "train_normalized": train_normalized_folder,
        "train_means": train_means_folder,
        "train_vars": train_vars_folder,
        "test_params": test_params_folder,
        "test_normalized": test_normalized_folder,
        "test_means": test_means_folder,
        "test_vars": test_vars_folder,
    }


def init_folders_for_mean_layer(root_folder: str) -> dict:
    mean_layer_folder = init_folder(os.path.join(root_folder, "mean_layer"))
    mean_layer_filename = os.path.join(mean_layer_folder, "mean_layer.pkl")
    mean_layer_results_filename = os.path.join(mean_layer_folder, "results.txt")
    mean_layer_preds_folder = init_folder(
        os.path.join(mean_layer_folder, f"test_mean_layer_preds")
    )
    return {
        "mean_layer": mean_layer_folder,
        "mean_layer_filename": mean_layer_filename,
        "mean_layer_results_filename": mean_layer_results_filename,
    }


def init_folders_for_dl(root_folder: str) -> dict:
    dl_model_folder = init_folder(os.path.join(root_folder, "dl_model"))
    dl_model_filename = os.path.join(dl_model_folder, "dl_model.pkl")
    dl_model_results_folder = init_folder(os.path.join(dl_model_folder, "results"))
    return {
        "dl_model": dl_model_folder,
        "dl_model_filename": dl_model_filename,
        "dl_model_results": dl_model_results_folder,
    }


def run_gas_experiment(
    dataset_name: str,
    dataset_type: str,
    dataset_params: dict,
    root_folder_name: str,
    normalizer_name: str,
    normalizer_inital_guesses: np.ndarray,
    normalizer_bounds: tuple,
    mean_layer_name: str,
    dl_model_library: str,
    dl_model_name: str,
    dataset_file_folder: str | None = None,
    normalizer_params: dict = {},
    mean_layer_params: dict = {},
    dl_model_params: dict = {},
    n_training_samples: int = 5000,
    n_test_samples: int = 1000,
    probabilistic: bool = False,
) -> tuple:
    # INITIALIZE ROOT FOLDERS
    root_folder = init_folder(root_folder_name)

    # INITIALIZE DATA MANAGER
    multivariate = dataset_params["multivariate"]
    data_manager = GluonTSDataManager(dataset_name, multivariate, dataset_file_folder, dataset_params["prediction_length"])

    # if the dataset is synthetic, we must save it
    if dataset_type == "synthetic":
        raise NotImplementedError
        """
        train_dataset_filename = os.path.join(root_folder, "train_dataset.pkl")
        test_dataset_filename = os.path.join(root_folder, "test_dataset.pkl")
        # if it already exists, we load it (even if we computed it)
        if os.path.exists(train_dataset_filename):
            with open(train_dataset_filename, "rb") as f:
                train_orig_dataset = pickle.load(f)
            with open(test_dataset_filename, "rb") as f:
                test_orig_dataset = pickle.load(f)
        # otherwise we save it
        else:
            with open(train_dataset_filename, "wb") as f:
                pickle.dump(train_orig_dataset, f)
            with open(test_dataset_filename, "wb") as f:
                pickle.dump(test_orig_dataset, f)
        """

    # NORMALIZATION PHASE
    # with this phase we will save
    # - normalizer initialization parameters
    # - normalizer best params, normalized_ts, means, vars for each ts for train and test
    normalizer_folders = init_folders_for_normalizer(root_folder)

    # this function computes and saves results and parameters from the normalization
    normalizer, processed_data = experiment_normalizer(
        normalizer_name,
        normalizer_params,
        data_manager.get_dataset_for_normalizer(),
        data_manager.context_length,
        normalizer_inital_guesses,
        normalizer_bounds,
        normalizer_folders,
    )

    # set the data for the next steps
    data_manager.set_data_from_normalizer(*processed_data)

    # MEAN LAYER PHASE
    # with this phase we will save
    # - mean layer initialization parameters
    # - trained mean layer
    # - score of the training phase
    # - mean_layer next point predictions for each time series in the test dataset

    mean_layer_folders = init_folders_for_mean_layer(root_folder)

    # this function computes and saves results and parameters from the mean layer
    if mean_layer_name == "linear":
        mean_layer = experiment_mean_layer_linear(
            data_manager.get_dataset_for_linear_mean_layer(
                n_training_samples, n_test_samples
            ),
            mean_layer_params,
            mean_layer_folders,
        )
    elif mean_layer_name == "gas":
        mean_layer = normalizer
    else:
        raise ValueError(f"Unknown mean layer method: {mean_layer_name}")

    # DEEP LEARNING MODEL PHASE
    # with this phase we will save
    # - torch model initialization parameters
    # - trained torch model
    # - torch model results

    dl_folders = init_folders_for_dl(root_folder)

    if dl_model_library == "gluonts":
        experiment_gluonts(
            data_manager.n_features,
            data_manager.context_length,
            data_manager.prediction_length,
            data_manager.freq,
            data_manager.get_gluon_dataset_for_dl_layer(),
            mean_layer,
            dl_model_name,
            dl_model_params,
            dl_folders,
            probabilistic,
        )
    elif dl_model_library == "torch":
        experiment_torch(
            data_manager.n_features,
            data_manager.context_length,
            data_manager.prediction_length,
            data_manager.get_torch_dataset_for_dl_layer(
                n_training_samples, n_test_samples
            ),
            mean_layer,
            dl_model_name,
            dl_model_params,
            dl_folders,
        )
    else:
        raise ValueError(f"Unknown deep learning library: {dl_model_library}")

    return (
        data_manager.n_features,
        data_manager.context_length,
        data_manager.prediction_length,
        data_manager.freq,
        data_manager.get_gluon_dataset_for_dl_layer(),
        mean_layer,
        dl_model_name,
        dl_model_params,
        dl_folders,
    )
