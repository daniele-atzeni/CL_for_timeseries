from gluonts.mx import Trainer
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.common import ListDataset
from gluonts.mx.distribution import StudentTOutput, MultivariateGaussianOutput

import mxnet as mx
import torch
from torch.utils.data import TensorDataset, DataLoader

from my_models.gluonts_models.simple_feedforward._estimator import (
    SimpleFeedForwardEstimator as FF_gluonts,
)
from my_models.gluonts_models.simple_feedforward_multivariate._estimator import (
    SimpleFeedForwardEstimator as FF_gluonts_multivariate,
)
from my_models.pytorch_models.simple_feedforward import FFNN as FF_torch

import os
import pickle
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import prepare_dataset_for_torch_model


def experiment_gluonts(
    n_features: int,
    context_length: int,
    prediction_length: int,
    train_dataset: list[np.ndarray],
    train_means: list[np.ndarray],
    train_vars: list[np.ndarray],
    test_dataset: list[np.ndarray],
    test_means: list[np.ndarray],
    test_vars: list[np.ndarray],
    weights: np.ndarray,
    bias: np.ndarray,
    dl_model_name: str,
    dl_model_params: dict,
    dl_model_folder: str,
    dl_model_filename: str,
    results_folder: str,
    freq: str,
    train_starts: list[pd.Timestamp],
    test_starts: list[pd.Timestamp],
) -> None:
    # retrieve initialization parameters
    estimator_parameters = dl_model_params["main_model"]
    trainer_parameters = dl_model_params["training"]
    predictor_parameters = dl_model_params["prediction"]
    evaluator_parameters = dl_model_params["evaluation"]
    # GLUONTS DATASET INITIALIZATION
    # we must create a new GluonTS dataset wirh normalized time series and means
    # and vars as new features.
    # feat_dynamic_real will be a 2D array of shape (2 * n_features, n_samples)
    # the first n_feature rows are the means, the second are the vars.

    if n_features == 1:
        gluonts_train_dataset = ListDataset(
            [
                {
                    "target": el.squeeze(),  # if univariate, this must be 1D
                    "start": start,
                    "feat_dynamic_real": np.concatenate((mean, var), axis=1).T,
                }
                for el, start, mean, var in zip(
                    train_dataset, train_starts, train_means, train_vars
                )
            ],
            freq=freq,
        )
        gluonts_test_dataset = ListDataset(
            [
                {
                    "target": el.squeeze(),  # if univariate, this must be 1D
                    "start": start,
                    "feat_dynamic_real": np.concatenate((mean, var), axis=1).T,
                }
                for el, start, mean, var in zip(
                    test_dataset, test_starts, test_means, test_vars
                )
            ],
            freq=freq,
        )
    else:
        gluonts_train_dataset = ListDataset(
            [
                {
                    "target": el.T,
                    "start": start,
                    "feat_dynamic_real": np.concatenate((mean, var), axis=1).T,
                }
                for el, start, mean, var in zip(
                    train_dataset, train_starts, train_means, test_vars
                )
            ],
            freq=freq,
            one_dim_target=False,
        )
        gluonts_test_dataset = ListDataset(
            [
                {
                    "target": el.T,
                    "start": start,
                    "feat_dynamic_real": np.concatenate((mean, var), axis=1).T,
                }
                for el, start, mean, var in zip(
                    test_dataset, test_starts, test_means, test_vars
                )
            ],
            freq=freq,
            one_dim_target=False,
        )

    # ESTIMATOR INITIALIZATION
    # we have to initialize the mean linear layer first
    print("Initializing the mean linear layer...")
    mean_layer = mx.gluon.nn.HybridSequential()
    mean_layer.add(
        mx.gluon.nn.Dense(
            units=prediction_length,
            weight_initializer=mx.init.Constant(weights),
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

    # estimator initialization
    print("Initializing the estimator...")
    trainer = Trainer(**trainer_parameters)
    if dl_model_name == "feedforward":
        if n_features == 1:
            estimator = FF_gluonts(
                mean_layer,
                StudentTOutput(),
                prediction_length=prediction_length,
                context_length=context_length,
                trainer=trainer,
                **estimator_parameters,
            )
        else:
            estimator = FF_gluonts_multivariate(
                mean_layer,
                MultivariateGaussianOutput(dim=n_features),
                prediction_length=prediction_length,
                context_length=context_length,
                trainer=trainer,
                **estimator_parameters,
            )
    else:
        raise ValueError(f"Unknown estimator name: {dl_model_name}")

    # TRAIN THE ESTIMATOR
    print("Training the estimator...")
    predictor = estimator.train(gluonts_train_dataset)
    # gluonts is not unbound because we checked the length of the dataset
    print("Done.")

    # EVALUATE IT
    print("Evaluating the estimator...")
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=gluonts_test_dataset,  # test dataset
        predictor=predictor,  # predictor
        **predictor_parameters,
    )
    print("Done.")
    # process results
    forecasts = list(forecast_it)
    tss = list(ts_it)
    evaluator = Evaluator(**evaluator_parameters)
    agg_metrics, item_metrics = evaluator(tss, forecasts)
    print(json.dumps(agg_metrics, indent=4))
    print(item_metrics.head())

    # SAVE EVERYTHING
    # save initialization parameters
    with open(os.path.join(dl_model_folder, "init_params.json"), "w") as f:
        json.dump(dl_model_params, f)
    # save evaluator and its results
    with open(dl_model_filename, "wb") as f:
        pickle.dump(evaluator, f)
    # save agg_metrics as json and item_metrics as csv
    with open(os.path.join(results_folder, "agg_metrics.json"), "w") as f:
        json.dump(agg_metrics, f)
    item_metrics.to_csv(os.path.join(results_folder, "item_metrics.csv"))


def experiment_torch(
    n_features: int,
    context_length: int,
    prediction_length: int,
    norm_train_dataset: list[np.ndarray],
    train_means: list[np.ndarray],
    train_vars: list[np.ndarray],
    norm_test_dataset: list[np.ndarray],
    test_means: list[np.ndarray],
    test_vars: list[np.ndarray],
    weights: np.ndarray,
    bias: np.ndarray,
    dl_model_name: str,
    dl_model_params: dict,
    dl_model_folder: str,
    dl_model_filename: str,
    results_folder: str,
    n_training_samples: int,
    n_test_samples: int,
) -> None:
    # retrieve initialization parameters
    model_parameters = dl_model_params["main_model"]
    training_parameters = dl_model_params["training"]
    prediction_parameters = dl_model_params["prediction"]
    evaluation_parameters = dl_model_params["evaluation"]
    # PYTORCH DATASET INITIALIZATION
    train_x, train_mean, train_y = prepare_dataset_for_torch_model(
        norm_train_dataset,
        train_means,
        context_length,
        prediction_length,
        n_training_samples,
    )
    test_x, test_mean, test_y = prepare_dataset_for_torch_model(
        norm_test_dataset, test_means, context_length, prediction_length, n_test_samples
    )
    # x: (n_samples, context_length, n_features)
    # mean: (n_samples, context_length * n_features)
    # y: (n_samples, prediction_length, n_features)

    train_dataset = TensorDataset(train_x, train_mean, train_y)
    train_dataloader = DataLoader(
        train_dataset, batch_size=training_parameters["batch_size"]
    )

    # MODEL INITIALIZATION
    # we have to initialize the mean linear layer first
    print("Initializing the mean linear layer...")
    mean_layer = torch.nn.Linear(
        context_length * n_features, prediction_length * n_features
    )
    mean_layer.weight.data = torch.from_numpy(weights).float()
    mean_layer.bias.data = torch.from_numpy(bias).float()
    # freeze the parameters
    for param in mean_layer.parameters():
        param.requires_grad = False
    print("Done.")

    # model initialization
    print("Initializing the model...")
    if dl_model_name == "feedforward":
        model = FF_torch(
            mean_layer,
            n_features=n_features,
            context_length=context_length,
            prediction_length=prediction_length,
            **model_parameters,
        )
    else:
        raise ValueError(f"Unknown model name: {dl_model_name}")

    # TRAIN THE MODEL
    print("Training the model...")
    if training_parameters["loss"] == "mse":
        loss = torch.nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss name: {training_parameters['loss']}")
    if training_parameters["optimizer"] == "adam":
        lr = training_parameters["learning_rate"]
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer name: {training_parameters['optimizer']}")

    model.train()
    for epoch in tqdm(range(training_parameters["epochs"]), unit="epoch"):
        for train_x, train_mean, train_y in train_dataloader:
            optimizer.zero_grad()
            output = model(train_x, train_mean)
            loss_value = loss(output, train_y)
            loss_value.backward()
            optimizer.step()
    print("Done.")

    # EVALUATE IT
    print("Evaluating the model...")
    model.eval()
    pred_y = model(test_x, test_mean)
    loss_value = loss(pred_y, test_y).item()
    print("Done.")

    # SAVE EVERYTHING
    # save initialization parameters
    with open(os.path.join(dl_model_folder, "init_params.json"), "w") as f:
        json.dump(dl_model_params, f)
    # save the model and its results
    with open(dl_model_filename, "wb") as f:
        pickle.dump(model, f)
    # save agg_metrics as json and item_metrics as csv
    with open(os.path.join(results_folder, "metrics.txt"), "w") as f:
        f.write(str(loss_value))


def experiment_dl_model(
    library: str,
    n_features: int,
    context_length: int,
    prediction_length: int,
    train_dataset: list[np.ndarray],
    train_means: list[np.ndarray],
    train_vars: list[np.ndarray],
    test_dataset: list[np.ndarray],
    test_means: list[np.ndarray],
    test_vars: list[np.ndarray],
    weights: np.ndarray,
    bias: np.ndarray,
    dl_model_name: str,
    dl_model_params: dict,
    dl_model_folder: str,
    dl_model_filename: str,
    results_folder: str,
    freq: str | None = None,
    train_starts: list[pd.Timestamp] | None = None,
    test_starts: list[pd.Timestamp] | None = None,
    n_training_samples: int | None = None,
    n_test_samples: int | None = None,
):
    if library == "torch":
        assert (
            n_training_samples is not None
        ), "n_training_samples must be specified for torch"
        assert n_test_samples is not None, "n_test_samples must be specified for torch"
        experiment_torch(
            n_features,
            context_length,
            prediction_length,
            train_dataset,
            train_means,
            train_vars,
            test_dataset,
            test_means,
            test_vars,
            weights,
            bias,
            dl_model_name,
            dl_model_params,
            dl_model_folder,
            dl_model_filename,
            results_folder,
            n_training_samples,
            n_test_samples,
        )

    elif library == "gluonts":
        assert freq is not None, "freq must be specified for gluonts"
        assert train_starts is not None, "train_starts must be specified for gluonts"
        assert test_starts is not None, "test_starts must be specified for gluonts"
        experiment_gluonts(
            n_features,
            context_length,
            prediction_length,
            train_dataset,
            train_means,
            train_vars,
            test_dataset,
            test_means,
            test_vars,
            weights,
            bias,
            dl_model_name,
            dl_model_params,
            dl_model_folder,
            dl_model_filename,
            results_folder,
            freq,
            train_starts,
            test_starts,
        )
    else:
        raise ValueError(f"Unknown library: {library}")
