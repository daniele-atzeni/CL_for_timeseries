from gluonts.dataset.repository import get_dataset
from gluonts.mx import Trainer
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.common import ListDataset

import mxnet as mx

from my_simple_feedforward._estimator import SimpleFeedForwardEstimator
from normalizer import GASComplexGaussian

import os
import pickle
import json

from sklearn import linear_model
import numpy as np


def prepare_dataset_for_mean_layer(
    means: list[np.ndarray],
    dataset: list[np.ndarray],
    context_length: int,
    prediction_length: int,
    n_test_sample_per_ts: int,
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
    These ts windows must be reshape to be 2D arrays.
    """
    len_dataset = len(dataset)
    assert len_dataset == len(means), "Dataset and means must have the same length"
    n_el = len_dataset * n_test_sample_per_ts
    n_features = 1 if len(dataset[0].shape) == 1 else dataset[0].shape[1]

    mean_layer_x = np.empty((n_el, context_length * n_features))
    mean_layer_y = np.empty((n_el, prediction_length * n_features))

    # computing starting indices
    start_indices = []
    for ts in dataset:
        start_indices.append(
            np.random.randint(
                low=0,
                high=ts.shape[0] - context_length - prediction_length,
                size=n_test_sample_per_ts,
            )
        )

    # slice and fill the arrays
    for i, (ts, mean_ts, start_idxs) in enumerate(zip(dataset, means, start_indices)):
        for j, start_idx in enumerate(start_idxs):
            mean_window_x = mean_ts[start_idx : start_idx + context_length]
            mean_layer_x[i * n_test_sample_per_ts + j] = mean_window_x.reshape(
                context_length * n_features
            )
            mean_window_y = ts[
                start_idx
                + context_length : start_idx
                + context_length
                + prediction_length
            ]
            mean_layer_y[i * n_test_sample_per_ts + j] = mean_window_y.reshape(
                prediction_length * n_features
            )

    return mean_layer_x, mean_layer_y


def main(
    dataset_name: str,
    n_training_samples_per_ts: int,
    n_test_sample_per_ts: int,
    root_folder: str,
) -> None:
    # INITIALIZE VARIOUS FOLDERS
    if not os.path.exists(root_folder):
        os.mkdir(root_folder)
    dataset_folder = os.path.join(root_folder, dataset_name)
    if not os.path.exists(dataset_folder):
        os.mkdir(dataset_folder)
    warm_up_results_folder = os.path.join(dataset_folder, "warm_up_results")
    if not os.path.exists(warm_up_results_folder):
        os.mkdir(warm_up_results_folder)
    norm_parameters_folder = os.path.join(warm_up_results_folder, "normalizer_params")
    if not os.path.exists(norm_parameters_folder):
        os.mkdir(norm_parameters_folder)
    norm_ts_folder = os.path.join(warm_up_results_folder, "normalized_ts")
    if not os.path.exists(norm_ts_folder):
        os.mkdir(norm_ts_folder)
    means_folder = os.path.join(warm_up_results_folder, "means")
    if not os.path.exists(means_folder):
        os.mkdir(means_folder)
    vars_folder = os.path.join(warm_up_results_folder, "vars")
    if not os.path.exists(vars_folder):
        os.mkdir(vars_folder)
    mean_layer_preds_folder = os.path.join(dataset_folder, "mean_layer_predictions")
    if not os.path.exists(mean_layer_preds_folder):
        os.mkdir(mean_layer_preds_folder)
    estimator_preds_folder = os.path.join(dataset_folder, "estimator_predictions")
    if not os.path.exists(estimator_preds_folder):
        os.mkdir(estimator_preds_folder)

    # GET THE DATASET
    print("Getting the dataset...")
    dataset = get_dataset(dataset_name)
    print("Done.")

    # WE SUPPOSE THAT THE DATASET HAS TIME SERIES OF EQUAL LENGTHS
    assert len(set([el["target"].shape[0] for el in dataset.train])) == 1, (
        "Time series of different lengths in the train dataset. "
        "This is not supported by the normalizer."
    )

    # INITIALIZE SOME PARAMETERS
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

    # We must produce a new dataset, with normalized time series and means as new features
    # we are sure that downloaded GluonTS datasets have no other features
    train_dataset = [el["target"] for el in dataset.train]
    # we will use starts for the initialization of a new GluonTS dataset
    train_starts = [el["start"] for el in dataset.train]
    assert dataset.test is not None, "Test dataset cannot be None"
    test_dataset = [el["target"] for el in dataset.test]
    test_starts = [el["start"] for el in dataset.test]

    # INITIALIZE AND WARM UP THE NORMALIZER
    normalizer = GASComplexGaussian()
    # normalizer is able to compute
    # - ideal initial guesses and static parameters of the normalizer for each time series in the dataset
    # - normalized time series, means, and variances for each time series in the dataset
    # it always expects a list of arrays as input
    print("Warming up train dataset...")
    train_normalizer_params = normalizer.warm_up(train_dataset)
    print("Warming up test dataset...")
    test_normalizer_params = normalizer.warm_up(test_dataset)
    print("Done.")
    # save the normalizer parameters with pickle
    print("Saving normalizer parameters...")
    for i, el in enumerate(train_normalizer_params):
        with open(
            os.path.join(norm_parameters_folder, f"train_{i}_normalizer_params.pkl"),
            "wb",
        ) as f:
            pickle.dump(train_normalizer_params, f)
    for i, el in enumerate(test_normalizer_params):
        with open(
            os.path.join(norm_parameters_folder, f"test_{i}_normalizer_params.pkl"),
            "wb",
        ) as f:
            pickle.dump(test_normalizer_params, f)

    # NORMALIZE THE DATASET
    print("Normalizing train dataset...")
    norm_train_dataset, train_means, train_vars = normalizer.normalize(
        train_dataset, train_normalizer_params
    )
    print("Normalizing test dataset...")
    norm_test_dataset, test_means, test_vars = normalizer.normalize(
        test_dataset, test_normalizer_params
    )
    print("Done.")
    # save normalized_test_dataset, means and vars. They are list of np.arrays
    # we save only test ones because they are a superset of train ones
    print("Saving normalized test dataset, means and vars...")
    for i, el in enumerate(norm_test_dataset):
        np.save(os.path.join(norm_ts_folder, f"test_ts_{i}_normalized.npy"), el)
    for i, el in enumerate(test_means):
        np.save(os.path.join(means_folder, f"test_ts_{i}_means.npy"), el)
    for i, el in enumerate(test_vars):
        np.save(os.path.join(vars_folder, f"test_ts_{i}_vars.npy"), el)

    # LET'S START THE PREPARATION FOR THE MEAN LAYER
    ## CREATE THE DATASET
    print("Preparing the train dataset for the mean linear layer...")
    mean_layer_train_x, mean_layer_train_y = prepare_dataset_for_mean_layer(
        train_means,
        train_dataset,
        context_length,
        prediction_length,
        n_training_samples_per_ts,
    )
    print("Preparing the test dataset for the mean linear layer...")
    mean_layer_test_x, mean_layer_test_y = prepare_dataset_for_mean_layer(
        test_means,
        test_dataset,
        context_length,
        prediction_length,
        n_test_sample_per_ts,
    )
    print("Done.")

    ## FIT THE REGRESSOR
    print("Fitting the mean linear layer...")
    regr = linear_model.LinearRegression()
    regr.fit(mean_layer_train_x, mean_layer_train_y)
    ## EVALUATE THE REGRESSOR
    print(
        f"Score of the mean linear layer: {regr.score(mean_layer_test_x, mean_layer_test_y)}"
    )
    weight = regr.coef_
    bias = regr.intercept_
    # save the regressor
    with open(os.path.join(dataset_folder, "mean_layer_regressor.pkl"), "wb") as f:
        pickle.dump(regr, f)

    ## COMPUTE LINEAR PREDICTIONS FOR TEST SET AND SAVE THEM
    print("Computing linear predictions for test set...")
    mean_predictions = []
    for i, mean in enumerate(test_means):
        ts_predictions = np.empty_like(mean)
        ts_predictions[:context_length] = mean[:context_length]
        for j in range(context_length, mean.shape[0]):
            pred = regr.predict(mean[j - context_length : j].reshape(1, -1))
            ts_predictions[j] = pred[0][0]  # we predict a single value per iteration
        mean_predictions.append(ts_predictions)
    # save mean predictions
    for i, el in enumerate(mean_predictions):
        np.save(
            os.path.join(mean_layer_preds_folder, f"test_ts_{i}_mean_layer_preds.npy"),
            el,
        )
    print("Done.")

    # NOW WE HAVE TO PREPARE FOR THE ESTIMATOR TRAINING PROCESS
    ## WE MUST FIRST PREPARE THE DATASET
    # we must create a new GluonTS dataset wirh normalized time series and means as new features
    new_train_dataset = ListDataset(
        [
            {
                "target": el,
                "start": start,
                "feat_dynamic_real": mean.reshape((n_features, -1)),
            }  # GluonTS expect 2D arrays as dynamic_features of shape (n_features, length)
            for el, start, mean in zip(norm_train_dataset, train_starts, train_means)
        ],
        freq=freq,
    )
    new_test_dataset = ListDataset(
        [
            {
                "target": el,
                "start": start,
                "feat_dynamic_real": mean.reshape((n_features, -1)),
            }  # GluonTS expect 2D arrays as dynamic_features of shape (n_features, length)
            for el, start, mean in zip(norm_test_dataset, test_starts, test_means)
        ],
        freq=freq,
    )

    ## INITIALIZE THE ESTIMATOR USING WEIGHTS AND BIAS FROM THE LINEAR REGRESSOR
    ### INITIALIZE MXNET LINEAR MEAN LAYER
    print("Initializing the mean linear layer...")
    mean_layer = mx.gluon.nn.HybridSequential()
    mean_layer.add(
        mx.gluon.nn.Dense(
            units=prediction_length,
            weight_initializer=mx.init.Constant(weight),
            bias_initializer=mx.init.Constant(bias),  # type: ignore # bias is a numpy array, don't know the reasons for this typing error
        )
    )
    mean_layer.add(
        mx.gluon.nn.HybridLambda(
            lambda F, o: F.reshape(
                o, (-1, prediction_length)
            )  # no need for that but just to be sure
        )
    )
    # freeze the parameters
    for param in mean_layer.collect_params().values():
        param.grad_req = "null"

    ### INITIALIZE THE ESTIMATOR
    print("Initializing the estimator...")
    num_hidden_dimensions = [128, 17]
    estimator = SimpleFeedForwardEstimator(
        mean_layer,
        num_hidden_dimensions=num_hidden_dimensions,
        prediction_length=prediction_length,
        context_length=context_length,
        trainer=Trainer(epochs=5, learning_rate=1e-3, num_batches_per_epoch=100),
    )
    ## TRAIN
    print("Training the estimator...")
    predictor = estimator.train(new_train_dataset)
    print("Done.")
    ## EVALUATE
    print("Evaluating the estimator...")
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=new_test_dataset,  # test dataset
        predictor=predictor,  # predictor
        num_samples=100,  # number of sample paths we want for evaluation
    )
    print("Done.")

    forecasts = list(forecast_it)
    tss = list(ts_it)

    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    agg_metrics, item_metrics = evaluator(tss, forecasts)

    print(json.dumps(agg_metrics, indent=4))
    print(item_metrics.head())


if __name__ == "__main__":
    DATASET_NAME = "traffic"
    N_TRAINING_SAMPLES_PER_TS = 100
    N_TEST_SAMPLES_PER_TS = 100
    ROOT_FOLDER = "UNIVARIATE_RESULTS"

    main(DATASET_NAME, N_TRAINING_SAMPLES_PER_TS, N_TEST_SAMPLES_PER_TS, ROOT_FOLDER)
