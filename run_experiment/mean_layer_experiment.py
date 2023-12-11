import os
import pickle
import json

from sklearn import linear_model
import numpy as np


def experiment_mean_layer(
    dataset: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    mean_layer_name: str,
    mean_layer_params: dict,
    folders: dict,
) -> None:
    (
        mean_layer_train_x,
        mean_layer_train_y,
        mean_layer_test_x,
        mean_layer_test_y,
    ) = dataset
    # REGRESSOR INITIALIZATION
    if mean_layer_name == "linear":
        mean_layer = linear_model.LinearRegression(**mean_layer_params)
    else:
        raise ValueError(f"Unknown regressor class: {mean_layer_name}")

    # FIT THE REGRESSOR AND EVALUATE IT
    print("Fitting the mean linear layer...")
    mean_layer.fit(mean_layer_train_x, mean_layer_train_y)
    #    evaluate
    results = mean_layer.score(mean_layer_test_x, mean_layer_test_y)
    print(f"Score of the mean linear layer: {results}")

    # SAVE EVERYTHING
    # save initialization parameters
    with open(os.path.join(folders["mean_layer"], "init_params.json"), "w") as f:
        json.dump(mean_layer_params, f)
    # save_results as a text file
    with open(folders["mean_layer_results_filename"], "w") as f:
        f.write(f"Score of the mean linear layer: {results}")
    # save the regressor
    with open(folders["mean_layer_filename"], "wb") as f:
        pickle.dump(mean_layer, f)
    # save mean predictions
    print("Done.")
