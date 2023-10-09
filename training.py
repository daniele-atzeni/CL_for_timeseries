from normalizer import Normalizer
from denormalizer import Denormalizer, SumDenormalizer, ConcatDenormalizer

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader

from gluonts.dataset import Dataset
from gluonts.model.estimator import Estimator
from gluonts.model.predictor import Predictor

from gluonts.transform import AdhocTransform

from gluonts.mx import SimpleFeedForwardEstimator, Trainer
import numpy as np


def train_gluonts_model(
    model: Estimator,
    normalizer: Normalizer,
    denormalizer: Denormalizer,
    ts: Dataset,
    num_epochs: int,
    estimator_params: dict,
) -> Predictor:
    estimator = model.__init__(**estimator_params)
    # convert the ts to a tensor dataset
    tensor_ts = torch.stack([torch.from_numpy(el["target"]) for el in ts.train])
    # retrieve gas means and vars
    mus = normalizer.mus
    vars = normalizer.vars
    assert isinstance(mus, Tensor) and isinstance(
        vars, Tensor
    )  # this is for sure a GAS normalizer, mus and vars are tensors or None
    # normalize the time series
    tensor_ts = (tensor_ts - mus) / (torch.sqrt(vars) + normalizer.eps)
    # for each feature of the time series and for each timestamp, we must create a new set of features containing the predictions of the mean layer for that feature
    # we must know the input length of the mean layer
    len_to_pred, input_length = estimator.prediction_length, estimator.context_length
    n_feat = tensor_ts.shape[1]
    new_features = torch.zeros((tensor_ts.shape[0], len_to_pred, n_feat))
    # first input_length predictions are 0
    new_features[:input_length, :, :] = 0
    for i in range(mus.shape[0] - input_length):
        # input of the mean layer is (batch, input_length, n_features)
        # we must change only the train, that does not include last len_to_pred elements
        mean_input = mus[i : i + input_length].unsqueeze(0)
        prediction = denormalizer.process_mus(mean_input).squeeze()  # type: ignore we are sure this is a nn.Module of the GASDenormalizer
        # prediction has shape (len_to_pred, n_features)
        new_features[i, :, :] = prediction
    # for each feature, we must create len_to_pred new timeseries. In the j-th new timeseries there is the j-th prediction for that feature
    new_feat_dict_list = [
        {f"pred_{j}": new_features[:, j, i] for j in range(len_to_pred)}
        for i in range(n_feat)
    ]
    # now we have to include these new dictionaries into the original dataset ts
    for i, el in enumerate(ts.train):
        el.update(new_feat_dict_list[i])

    # define the transformation that subtracts the prediction of the mean layer from the target
    def add_means_feat_and_subtract(x):
        x["future_target"] = x["future_target"] - np.array(
            [x[f"pred_{i}"][x["start_future"]] for i in range(len_to_pred)]
        )
        return x

    train_ds = AdhocTransform(add_means_feat_and_subtract)(
        iter(ts.train), is_train=True
    )
    return estimator.train(train_ds)


def train_torch_model(
    normalizer: Normalizer,
    model: nn.Module,
    denormalizer: Denormalizer,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    criterion,
    optimizer,
    num_epochs: int,
) -> None:
    for epoch in range(num_epochs):
        epoch_loss = 0
        for ts_indices, ts_x, ts_y in train_dataloader:
            # training loop
            optimizer.zero_grad()

            # retrieve/compute mus and vars
            norm_ts, mus, _ = normalizer.normalize(ts_indices, ts_x)

            vars = torch.ones_like(mus)

            proc_ts = model(norm_ts)
            # compute output
            out = denormalizer(proc_ts, mus, vars)

            #####
            # we have to reshape ts_y to be of shape (batch_size, -1)
            ts_y = ts_y.reshape((ts_y.shape[0], -1))
            #####

            loss = criterion(out, ts_y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # if (epoch + 1) % 100 == 0:
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] of the main model, Loss: {epoch_loss:.4f}"
        )
    return


def train_mean_encoder(
    normalizer: Normalizer,
    denormalizer: Denormalizer,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    criterion,
    optimizer,
    num_epochs: int,
) -> None:
    for epoch in range(num_epochs):
        epoch_loss = 0
        for ts_indices, ts_x, ts_y in train_dataloader:
            # training loop
            optimizer.zero_grad()

            # retrieve/compute mus and vars
            _, mus, _ = normalizer.normalize(ts_indices, ts_x)
            proc_ts = None
            vars = None
            # compute output
            out = denormalizer(proc_ts, mus, vars)

            #####
            # we have to reshape ts_y to be of shape (batch_size, -1)
            ts_y = ts_y.reshape((ts_y.shape[0], -1))
            #####

            loss = criterion(out, ts_y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # if (epoch + 1) % 100 == 0:
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] of mean encoder, Loss: {epoch_loss:.4f}"
        )
    return


def train_var_encoder(
    normalizer: Normalizer,
    model: nn.Module | Predictor,
    denormalizer: Denormalizer,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    criterion,
    optimizer,
    num_epochs: int,
) -> None:
    for epoch in range(num_epochs):
        epoch_loss = 0
        for ts_indices, ts_x, ts_y in train_dataloader:
            # training loop
            optimizer.zero_grad()

            # retrieve/compute mus and vars
            norm_ts, mus, vars = normalizer.normalize(ts_indices, ts_x)
            if isinstance(model, nn.Module):
                proc_ts = model(norm_ts)
            else:
                proc_ts = model.predict(norm_ts)
            # compute output
            out = denormalizer(proc_ts, mus, vars)

            #####
            # we have to reshape ts_y to be of shape (batch_size, -1)
            ts_y = ts_y.reshape((ts_y.shape[0], -1))
            #####

            loss = criterion(out, ts_y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # if (epoch + 1) % 100 == 0:
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] of var encoder, Loss: {epoch_loss:.4f}"
        )
    return


def train_concat(
    normalizer: Normalizer,
    model: nn.Module,
    denormalizer: Denormalizer,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    criterion,
    optimizer,
    num_epochs: int,
) -> None:
    for epoch in range(num_epochs):
        epoch_loss = 0
        for ts_indices, ts_x, ts_y in train_dataloader:
            # training loop
            optimizer.zero_grad()

            # retrieve/compute mus and vars
            norm_ts, mus, vars = normalizer.normalize(ts_indices, ts_x)
            proc_ts = model(norm_ts)

            # compute output
            out = denormalizer(proc_ts, mus, vars)

            #####
            # we have to reshape ts_y to be of shape (batch_size, -1)
            ts_y = ts_y.reshape((ts_y.shape[0], -1))
            #####

            loss = criterion(out, ts_y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # if (epoch + 1) % 100 == 0:
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] of mean encoder, Loss: {epoch_loss:.4f}"
        )
    return


def train(
    normalizer: Normalizer,
    model: nn.Module,
    denormalizer: Denormalizer,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    criterion,
    optimizer_class,
    optimizer_params: dict,
    num_epochs: int,
    main_ts: Tensor | None = None,
    estimator_params: dict | None = None,
) -> None:
    if isinstance(denormalizer, SumDenormalizer):
        # we must use three different optimizers
        mus_optim = optimizer_class(denormalizer.get_mus_params(), **optimizer_params)
        # phase one
        train_mean_encoder(
            normalizer,
            denormalizer,
            train_dataloader,
            test_dataloader,
            criterion,
            mus_optim,
            num_epochs,
        )
        # phase two is a gluonts training
        if isinstance(model, Estimator):
            assert main_ts is not None, "main_ts must be provided for GluonTS training"
            assert estimator_params is not None, "estimator_params must be provided"
            predictor = train_gluonts_model(
                model, normalizer, denormalizer, main_ts, num_epochs, estimator_params
            )  # model now is a predictor
        else:
            model_optim = optimizer_class(model.parameters(), **optimizer_params)
            train_torch_model(
                normalizer,
                model,
                denormalizer,
                train_dataloader,
                test_dataloader,
                criterion,
                model_optim,
                num_epochs,
            )
            predictor = model
        # phase 3
        var_optim = optimizer_class(denormalizer.get_vars_params(), **optimizer_params)
        train_var_encoder(
            normalizer,
            predictor,
            denormalizer,
            train_dataloader,
            test_dataloader,
            criterion,
            var_optim,
            num_epochs,
        )
        return
    elif isinstance(denormalizer, ConcatDenormalizer):
        assert not isinstance(
            model, Estimator
        ), "ConcatDenormalizer not supported with GluonTS"  # we can't modify the last layer of the estimator
        # we can use one optimizer with all the parameters and train everything in one phase
        params = (
            list(normalizer.parameters())
            + list(model.parameters())
            + list(denormalizer.parameters())
        )
        optimizer = optimizer_class(params, **optimizer_params)
        train_concat(
            normalizer,
            model,
            denormalizer,
            train_dataloader,
            test_dataloader,
            criterion,
            optimizer,
            num_epochs,
        )
    else:
        raise ValueError("Denormalizer type not supported")
