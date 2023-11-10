import os
import pickle
import json

from sklearn import linear_model
import numpy as np
from tqdm import tqdm

from utils import prepare_dataset_for_mean_layer, save_list_of_elements


def experiment_mean_layer(
    train_means: list[np.ndarray],
    train_dataset: list[np.ndarray],
    test_means: list[np.ndarray],
    test_dataset: list[np.ndarray],
    context_length: int,
    prediction_length: int,
    n_training_samples: int,
    n_test_samples: int,
    mean_layer_name: str,
    mean_layer_params: dict,
    mean_layer_folder: str,
    mean_layer_filename: str,
    mean_layer_results_filename: str,
    mean_layer_preds_folder: str,
) -> None:
    # REGRESSOR INITIALIZATION
    if mean_layer_name == "linear":
        mean_layer = linear_model.LinearRegression(**mean_layer_params)
    else:
        raise ValueError(f"Unknown regressor class: {mean_layer_name}")

    # CREATE THE MEAN LAYER DATASETS
    # this function will return
    # - x: np.arrays of shape (n_samples, context_length * n_features) using means
    # - y: np.arrays of shape (n_samples, prediction_length * n_features) using ts
    print("Preparing the train dataset for the mean linear layer...")
    mean_layer_train_x, mean_layer_train_y = prepare_dataset_for_mean_layer(
        train_means,
        train_dataset,
        context_length,
        prediction_length,
        n_training_samples,
    )
    print("Preparing the test dataset for the mean linear layer...")
    mean_layer_test_x, mean_layer_test_y = prepare_dataset_for_mean_layer(
        test_means,
        test_dataset,
        context_length,
        prediction_length,
        n_test_samples,
    )
    print("Done.")

    # FIT THE REGRESSOR AND EVALUATE IT
    print("Fitting the mean linear layer...")
    mean_layer.fit(mean_layer_train_x, mean_layer_train_y)
    #    evaluate
    results = mean_layer.score(mean_layer_test_x, mean_layer_test_y)
    print(f"Score of the mean linear layer: {results}")

    # COMPUTE LINEAR PREDICTIONS FOR TEST SET AND SAVE THEM
    print("Computing linear predictions for test set...")
    mean_predictions = []
    for i, mean in tqdm(enumerate(test_means), total=len(test_means), unit="ts"):
        ts_predictions = np.empty_like(mean)
        ts_predictions[:context_length] = mean[:context_length]
        ts_len, n_features = mean.shape
        for j in tqdm(range(context_length, ts_len), unit="step"):
            pred = mean_layer.predict(mean[j - context_length : j].reshape(1, -1))
            pred = pred.reshape(n_features, prediction_length)
            ts_predictions[j] = pred[:, 0]  # we predict a single value per iteration
        mean_predictions.append(ts_predictions)

    # SAVE EVERYTHING
    # save initialization parameters
    with open(os.path.join(mean_layer_folder, "init_params.json"), "w") as f:
        json.dump(mean_layer_params, f)
    # save_results as a text file
    with open(mean_layer_results_filename, "w") as f:
        f.write(f"Score of the mean linear layer: {results}")
    # save the regressor
    with open(mean_layer_filename, "wb") as f:
        pickle.dump(mean_layer, f)
    # save mean predictions
    save_list_of_elements(mean_layer_preds_folder, mean_predictions)
    print("Done.")
