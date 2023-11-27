import os
import pickle

import numpy as np

from utils import init_folder, load_list_of_elements, get_dataset_and_metadata
from run_experiment.normalizer_experiment import experiment_normalizer
from run_experiment.mean_layer_experiment import experiment_mean_layer
from run_experiment.dl_model_experiment import experiment_dl_model


def run_gas_experiment(
    dataset_name: str,
    dataset_type: str,
    dataset_generation_params: dict,
    root_folder_name: str,
    normalizer_name: str,
    normalizer_inital_guesses: np.ndarray,
    normalizer_bounds: tuple,
    mean_layer_name: str,
    dl_model_library: str,
    dl_model_name: str,
    normalizer_params: dict = {},
    mean_layer_params: dict = {},
    dl_model_params: dict = {},
    n_training_samples: int = 5000,
    n_test_samples: int = 1000,
    stop_after_normalizer: bool = False,
    stop_after_mean_layer: bool = False,
) -> None:
    # INITIALIZE ROOT FOLDERS
    root_folder = init_folder(root_folder_name)

    # GET THE DATASET AND INITIALIZE ITS PARAMETERS
    print("Getting the dataset...")
    (
        train_orig_dataset,
        test_orig_dataset,
        prediction_length,
        context_length,
        freq,
        n_features,
    ) = get_dataset_and_metadata(dataset_name, dataset_type, dataset_generation_params)
    print("Done.")

    # if the dataset is synthetic, we must save it
    if dataset_type == "synthetic":
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

    # CONVERT DATASET FOR NORMALIZER
    # We must produce a new dataset cause normalizers work on list of numpy arrays
    # NB we are sure that downloaded GluonTS datasets have no other features
    train_dataset = [el["target"].T for el in train_orig_dataset]  # (n_ts, n_samples)
    test_dataset = [el["target"].T for el in test_orig_dataset]
    # we also retrieve these values for initializing a new GluonTS dataset
    train_starts = [el["start"] for el in train_orig_dataset]
    test_starts = [el["start"] for el in test_orig_dataset]
    # gluonts datasets are univariate. Even in this case, we will unsqueeze last dim
    train_dataset = [
        np.expand_dims(el, -1) if len(el.shape) == 1 else el for el in train_dataset
    ]
    test_dataset = [
        np.expand_dims(el, -1) if len(el.shape) == 1 else el for el in test_dataset
    ]

    # NORMALIZATION PHASE
    # with this phase we will save
    # - normalizer initialization parameters
    # - normalizer best params, normalized_ts, means, vars for each ts for train and test
    normalizer_folder = init_folder(os.path.join(root_folder, normalizer_name))
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

    experiment_normalizer(
        normalizer_name,
        normalizer_params,
        train_dataset,
        test_dataset,
        normalizer_inital_guesses,
        normalizer_bounds,
        normalizer_folder,
        train_params_folder,
        train_normalized_folder,
        train_means_folder,
        train_vars_folder,
        test_params_folder,
        test_normalized_folder,
        test_means_folder,
        test_vars_folder,
    )
    if stop_after_normalizer:
        return

    # this function computes and saves results and parameters from the normalization
    # we have to load some data again for the next steps
    train_means = load_list_of_elements(train_means_folder)
    test_means = load_list_of_elements(test_means_folder)
    norm_train_dataset = load_list_of_elements(train_normalized_folder)
    norm_test_dataset = load_list_of_elements(test_normalized_folder)

    # MEAN LAYER PHASE
    # with this phase we will save
    # - mean layer initialization parameters
    # - trained mean layer
    # - score of the training phase
    # - mean_layer next point predictions for each time series in the test dataset
    mean_layer_folder = init_folder(os.path.join(root_folder, mean_layer_name))
    mean_layer_filename = os.path.join(mean_layer_folder, "mean_layer.pkl")
    mean_layer_results_filename = os.path.join(mean_layer_folder, "results.txt")
    mean_layer_preds_folder = init_folder(
        os.path.join(mean_layer_folder, f"test_mean_layer_preds")
    )

    experiment_mean_layer(
        train_means,
        train_dataset,
        test_means,
        test_dataset,
        context_length,
        prediction_length,
        n_training_samples,
        n_test_samples,
        mean_layer_name,
        mean_layer_params,
        mean_layer_folder,
        mean_layer_filename,
        mean_layer_results_filename,
        mean_layer_preds_folder,
    )
    if stop_after_mean_layer:
        return

    # this function computes and saves results and parameters from the mean layer
    # we have to load the regressor for the next step
    with open(mean_layer_filename, "rb") as f:
        regr = pickle.load(f)
    weights = regr.coef_
    bias = regr.intercept_

    # DEEP LEARNING MODEL PHASE
    # with this phase we will save
    # - torch model initialization parameters
    # - trained torch model
    # - torch model results
    dl_model_folder = init_folder(
        os.path.join(root_folder, f"{dl_model_library}_{dl_model_name}")
    )
    dl_model_filename = os.path.join(dl_model_folder, "dl_model.pkl")
    dl_model_results_folder = init_folder(os.path.join(dl_model_folder, "results"))

    experiment_dl_model(
        dl_model_library,
        n_features,
        context_length,
        prediction_length,
        norm_train_dataset,
        train_means,
        norm_test_dataset,
        test_means,
        weights,
        bias,
        dl_model_name,
        dl_model_params,
        dl_model_folder,
        dl_model_filename,
        dl_model_results_folder,
        freq,
        train_starts,
        test_starts,
        n_training_samples,
        n_test_samples,
    )
