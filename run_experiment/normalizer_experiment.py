from normalizer import GASNormalizer, GASComplexGaussian, GASTStudent

import os
import json

import numpy as np

from utils import save_list_of_elements


def experiment_normalizer(
    normalizer_name: str,
    normalizer_parameters: dict,
    datasets: tuple[list[np.ndarray], list[np.ndarray]],
    context_length: int,
    initial_guesses: np.ndarray,
    bounds: tuple,
    folders: dict,
) -> tuple[GASNormalizer, tuple]:
    # normalizer is able to compute
    # - ideal initial guesses and static parameters of the normalizer for each time series in the dataset
    # - normalized time series, means, and variances for each time series in the dataset
    # it always expects a list of arrays as input
    if normalizer_name == "gas_complex_gaussian":
        normalizer = GASComplexGaussian(**normalizer_parameters)
    elif normalizer_name == "gas_t_student":
        normalizer = GASTStudent(**normalizer_parameters)
    else:
        raise ValueError(f"Unknown normalizer class: {normalizer_name}")

    train_dataset, test_dataset = datasets

    print("Warming up train dataset...")
    use_context = normalizer.mean_strength != 0

    train_normalizer_params = normalizer.warm_up(
        train_dataset, context_length, initial_guesses, bounds, use_context=use_context
    )
    print("Warming up test dataset...")
    test_normalizer_params = normalizer.warm_up(
        test_dataset, context_length, initial_guesses, bounds, use_context=use_context
    )
    print("Done.")

    # NORMALIZE THE DATASET
    print("Normalizing train dataset...")
    norm_train_dataset, train_means, train_vars = normalizer.normalize(
        train_dataset, train_normalizer_params
    )

    # SAVE EVERYTHING
    # save normalizer initialization parameters as json
    with open(os.path.join(folders["normalizer"], "init_params.json"), "w") as f:
        json.dump(normalizer_parameters, f)
    # save the normalizer parameters with pickle
    print("Saving normalizer parameters...")
    save_list_of_elements(folders["train_params"], train_normalizer_params)
    save_list_of_elements(folders["test_params"], test_normalizer_params)
    # save normalized_train_dataset, means and vars. They are list of np.arrays
    print("Saving normalized train dataset, means and vars...")
    save_list_of_elements(folders["train_normalized"], norm_train_dataset)
    save_list_of_elements(folders["train_means"], train_means)
    save_list_of_elements(folders["train_vars"], train_vars)
    print("Done.")
    print("Normalizing test dataset...")
    norm_test_dataset, test_means, test_vars = normalizer.normalize(
        test_dataset, test_normalizer_params
    )
    print("Done.")
    # save normalized_test_dataset, means and vars. They are list of np.arrays
    print("Saving normalized test dataset, means and vars...")
    save_list_of_elements(folders["test_normalized"], norm_test_dataset)
    save_list_of_elements(folders["test_means"], test_means)
    save_list_of_elements(folders["test_vars"], test_vars)

    return normalizer, (
        train_means,
        train_vars,
        test_means,
        test_vars,
        train_normalizer_params,
    )
