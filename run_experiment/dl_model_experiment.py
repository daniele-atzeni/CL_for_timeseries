from gluonts.mx import Trainer
from gluonts.evaluation import (
    make_evaluation_predictions,
    Evaluator,
    MultivariateEvaluator,
)
from gluonts.dataset import DataEntry
from gluonts.mx.distribution import StudentTOutput, MultivariateGaussianOutput

import mxnet as mx
import torch
from torch.utils.data import TensorDataset, DataLoader

from my_models.gluonts_models.feedforward_linear_means._estimator import (
    SimpleFeedForwardEstimator as FF_gluonts,
)
from my_models.gluonts_models.multivariate_feedforward_linear_means._estimator import (
    SimpleFeedForwardEstimator as FF_gluonts_multivariate,
)
from my_models.pytorch_models.simple_feedforward import FFNN as FF_torch

from normalizer import GASNormalizer
from sklearn.linear_model import LinearRegression


import os
import pickle
import json

from tqdm import tqdm


def experiment_gluonts(
    n_features: int,
    context_length: int,
    prediction_length: int,
    dataset: tuple[list[DataEntry], list[DataEntry]],
    trained_mean_layer: LinearRegression | GASNormalizer,
    dl_model_name: str,
    dl_model_params: dict,
    folders: dict,
) -> None:
    # retrieve the dataset
    gluonts_train_dataset, gluonts_test_dataset = dataset
    # retrieve folders
    dl_model_folder = folders["dl_model"]
    dl_model_filename = folders["dl_model_filename"]
    results_folder = folders["dl_model_results"]
    # retrieve initialization parameters
    estimator_parameters = dl_model_params["main_model"]
    trainer_parameters = dl_model_params["training"]
    predictor_parameters = dl_model_params["prediction"]
    evaluator_parameters = dl_model_params["evaluation"]

    # ESTIMATOR INITIALIZATION
    # we have to initialize the mean linear layer first
    print("Initializing the mean linear layer...")
    if isinstance(trained_mean_layer, LinearRegression):
        mean_layer = mx.gluon.nn.HybridSequential()
        mean_layer.add(
            mx.gluon.nn.Dense(
                units=prediction_length * n_features,
                weight_initializer=mx.init.Constant(trained_mean_layer.coef_),
                bias_initializer=mx.init.Constant(trained_mean_layer.intercept_),  # type: ignore # bias is a numpy array, don't know the reasons for this typing error
            )
        )
        mean_layer.add(
            mx.gluon.nn.HybridLambda(
                lambda F, o: F.reshape(
                    o, (-1, prediction_length * n_features)
                )  # no need for that but just to be sure
            )
        )
    elif isinstance(trained_mean_layer, GASNormalizer):
        mean_layer = trained_mean_layer
    else:
        raise ValueError(
            f"Unknown mean layer type: {type(trained_mean_layer)} {trained_mean_layer}"
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
                n_features,
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
    if n_features == 1:
        evaluator = Evaluator(**evaluator_parameters)
    else:
        evaluator = MultivariateEvaluator(**evaluator_parameters)

    agg_metrics, item_metrics = evaluator(tss, forecasts)  # type: ignore # we are sure that tss is a list of DataFrame in multivariate case
    # print(json.dumps(agg_metrics, indent=4))
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
    datasets: tuple[TensorDataset, TensorDataset],
    trained_mean_layer: LinearRegression | GASNormalizer,
    dl_model_name: str,
    dl_model_params: dict,
    folders: dict,
) -> None:
    # retrieve folders
    dl_model_folder = folders["dl_model"]
    dl_model_filename = folders["dl_model_filename"]
    results_folder = folders["dl_model_results"]
    # retrieve initialization parameters
    model_parameters = dl_model_params["main_model"]
    training_parameters = dl_model_params["training"]
    prediction_parameters = dl_model_params["prediction"]
    evaluation_parameters = dl_model_params["evaluation"]
    # PYTORCH DATASET INITIALIZATION
    train_dataset, test_dataset = datasets

    train_dataloader = DataLoader(
        train_dataset, batch_size=training_parameters["batch_size"]
    )

    # MODEL INITIALIZATION
    # we have to initialize the mean linear layer first
    print("Initializing the mean linear layer...")
    if isinstance(trained_mean_layer, LinearRegression):
        mean_layer = torch.nn.Linear(
            context_length * n_features, prediction_length * n_features
        )
        mean_layer.weight.data = torch.from_numpy(trained_mean_layer.coef_).float()
        mean_layer.bias.data = torch.from_numpy(trained_mean_layer.intercept_).float()
        # freeze the parameters
        for param in mean_layer.parameters():
            param.requires_grad = False
        print("Done.")
    else:
        raise ValueError(
            f"Unknown mean layer type: {type(trained_mean_layer)} {trained_mean_layer}"
        )

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
        for train_x, train_mean, train_var, train_y in train_dataloader:
            optimizer.zero_grad()
            output = model(train_x, train_mean, train_var)
            loss_value = loss(output, train_y)
            loss_value.backward()
            optimizer.step()
    print("Done.")

    # EVALUATE IT
    print("Evaluating the model...")
    model.eval()
    running_loss = 0.0
    for test_x, test_mean, test_var, test_y in test_dataset:
        pred_y = model(test_x, test_mean, test_var)
        running_loss += loss(pred_y, test_y).item()
    loss_value = running_loss / len(test_dataset)
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
