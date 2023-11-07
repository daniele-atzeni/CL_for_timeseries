from normalizer import GASComplexGaussian

import os
import json

import numpy as np

from utils import save_list_of_elements


def experiment_normalizer(
    normalizer_name: str,
    normalizer_parameters: dict,
    train_dataset: list[np.ndarray],
    test_dataset: list[np.ndarray],
    normalizer_folder: str,
    train_params_folder: str,
    train_normalized_folder: str,
    train_means_folder: str,
    train_vars_folder: str,
    test_params_folder: str,
    test_normalized_folder: str,
    test_means_folder: str,
    test_vars_folder: str,
) -> None:
    # normalizer is able to compute
    # - ideal initial guesses and static parameters of the normalizer for each time series in the dataset
    # - normalized time series, means, and variances for each time series in the dataset
    # it always expects a list of arrays as input
    if normalizer_name == "gas_complex_gaussian":
        normalizer = GASComplexGaussian(**normalizer_parameters)
    else:
        raise ValueError(f"Unknown normalizer class: {normalizer_name}")

    print("Warming up train dataset...")
    train_normalizer_params = normalizer.warm_up(train_dataset)
    print("Warming up test dataset...")
    test_normalizer_params = normalizer.warm_up(test_dataset)
    print("Done.")

    # NORMALIZE THE DATASET
    print("Normalizing train dataset...")
    norm_train_dataset, train_means, train_vars = normalizer.normalize(
        train_dataset, train_normalizer_params
    )

    # SAVE EVERYTHING
    # save normalizer initialization parameters as json
    with open(os.path.join(normalizer_folder, "init_params.json"), "w") as f:
        json.dump(normalizer_parameters, f)
    # save the normalizer parameters with pickle
    print("Saving normalizer parameters...")
    save_list_of_elements(train_params_folder, train_normalizer_params)
    save_list_of_elements(test_params_folder, test_normalizer_params)
    # save normalized_train_dataset, means and vars. They are list of np.arrays
    print("Saving normalized train dataset, means and vars...")
    save_list_of_elements(train_normalized_folder, norm_train_dataset)
    save_list_of_elements(train_means_folder, train_means)
    save_list_of_elements(train_vars_folder, train_vars)
    print("Done.")
    print("Normalizing test dataset...")
    norm_test_dataset, test_means, test_vars = normalizer.normalize(
        test_dataset, test_normalizer_params
    )
    print("Done.")
    # save normalized_test_dataset, means and vars. They are list of np.arrays
    print("Saving normalized test dataset, means and vars...")
    save_list_of_elements(test_normalized_folder, norm_test_dataset)
    save_list_of_elements(test_means_folder, test_means)
    save_list_of_elements(test_vars_folder, test_vars)
