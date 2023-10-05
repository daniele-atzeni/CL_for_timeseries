import numpy as np

import torch
from torch import Tensor
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import MSELoss
from torch.optim import Adam

from normalizer import Normalizer, GASSimpleGaussian
from models import MyModel, FFNN
from denormalizer import Denormalizer, ConcatDenormalizer, SumDenormalizer
from utils import create_forecasting_dataset


def train_with_sum_denormalizer(
    normalizer: Normalizer,
    model: nn.Module,
    denormalizer: SumDenormalizer,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    criterion,
    optimizer_class,
    optimizer_params: dict,
    num_epochs: int,
) -> None:
    """
    We assume that our output function is in the form y = f(x)g(sigma) + h(mu)
    We will have three different optimizers, one for each function
    """
    optimizers = [
        optimizer_class(denormalizer.get_mus_params(), **optimizer_params),
        optimizer_class(model.parameters(), **optimizer_params),
        optimizer_class(denormalizer.get_vars_params(), **optimizer_params),
    ]

    for phase, optimizer in enumerate(optimizers):
        for epoch in range(num_epochs):
            epoch_loss = 0
            for ts_indices, ts_x, ts_y in train_dataloader:
                # training loop
                optimizer.zero_grad()

                norm_ts, mus, vars = normalizer.normalize(ts_indices, ts_x)
                proc_ts = model(norm_ts) if phase != 0 else Tensor(0)
                out = denormalizer(proc_ts, mus, vars, phase)

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
                f"Epoch [{epoch + 1}/{num_epochs}] of phase {phase}, Loss: {epoch_loss:.4f}"
            )
    return


def simple_train(
    normalizer: Normalizer,
    model: nn.Module,
    denormalizer: Denormalizer,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    criterion,
    optimizer_class,
    optimizer_params: dict,
    num_epochs: int,
) -> None:
    params = (
        list(normalizer.parameters())
        + list(model.parameters())
        + list(denormalizer.parameters())
    )
    optimizer = optimizer_class(params, **optimizer_params)

    for epoch in range(num_epochs):
        epoch_loss = 0
        for ts_indices, ts_x, ts_y in train_dataloader:
            # training loop
            optimizer.zero_grad()

            norm_ts, mus, vars = normalizer.normalize(ts_indices, ts_x)
            proc_ts = model(norm_ts)
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
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    return


def run_experiment(
    ts: torch.Tensor,
    perc_training_ts: float,
    ts_len: int,
    len_to_pred: int,
    n_train_sample: int,
    n_test_sample: int,
    normalizer_params: dict,
    model_params: dict,
    denormalizer_params: dict,
    criterion_name: str,
    optimizer_name: str,
    optimizer_params: dict = {},
    batch_size: int = 1,
    num_epochs: int = 5,
):
    """
    time_series is multivariate time series, represented as a tensor (length_ts, n_features)
    ts_normalizer is the normalizer
    main_model is the core model for processing time series (e.g. a feedforward model)
    """
    # ----------- DATASET PREPARATION --------

    # dividing the time series into training and test. How?
    # Until a certain time is train, then is test
    end_training_ind = int(perc_training_ts * ts.shape[0])
    train_ts = ts[:end_training_ind]
    test_ts = ts[end_training_ind:]

    # create the dataset
    # train_dataset = create_forecasting_dataset(train_norm_ts, train_mus, train_vars, TS_LEN, LEN_TO_PRED, N_SAMPLES)
    # test_dataset = create_forecasting_dataset(test_norm_ts, test_mus, test_vars, TS_LEN, LEN_TO_PRED, N_SAMPLES)
    train_dataset = create_forecasting_dataset(
        train_ts, ts_len, len_to_pred, n_train_sample
    )
    test_dataset = create_forecasting_dataset(
        test_ts, ts_len, len_to_pred, n_test_sample
    )
    # ----------------------------------------

    # ----------- INIT NORMALIZER ------------
    normalizer_class = normalizer_params.pop("class")
    normalizer = normalizer_class(**normalizer_params)
    # WARM UP PHASE: for GAS, compute static parameters and precompute means and vars of all the ts
    normalizer.warm_up(ts, train_ts)
    # ----------------------------------------

    # ----------- INIT MODEL -----------------
    model_class = model_params.pop("class")
    model = model_class(**model_params)
    # ----------------------------------------

    # ----------- INIT DENORMALIZER ----------
    denorm_class = denormalizer_params.pop("class")
    denormalizer = denorm_class(**denormalizer_params)
    # ----------------------------------------

    print(
        f"""Initialized normalizer, ts_model and denormalizer, with number of parameters: 
          Normalizer: {sum(p.numel() for p in normalizer.parameters() if p.requires_grad)}, 
          Main model : {sum(p.numel() for p in model.parameters() if p.requires_grad)},  
          Denormalizer : {sum(p.numel() for p in denormalizer.parameters() if p.requires_grad)}"""
    )

    # ----------- TRAINING -------------------
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset)
    if optimizer_name == "adam":
        optimizer_class = torch.optim.Adam
    else:
        raise NotImplementedError(f"Not implemented optimizer {optimizer_name}")

    if criterion_name == "mse":
        criterion = MSELoss()
    else:
        raise NotImplementedError(f"Not implemented criterion {criterion_name}")

    if isinstance(denormalizer, SumDenormalizer):
        train_with_sum_denormalizer(
            normalizer,
            model,
            denormalizer,
            train_dataloader,
            test_dataloader,
            criterion,
            optimizer_class,
            optimizer_params,
            num_epochs,
        )
    else:
        simple_train(
            normalizer,
            model,
            denormalizer,
            train_dataloader,
            test_dataloader,
            criterion,
            optimizer_class,
            optimizer_params,
            num_epochs,
        )
    return


if __name__ == "__main__":
    # initialize the time series
    ts = torch.load("traffic_hourly_train.pt")  # this is shape torch.Size([17448, 862])

    # dataset params
    PERC_TRAINING_TS = 0.75
    TS_LEN = 15
    LEN_TO_PRED = 15
    N_TRAIN_SAMPLES = 100
    N_TEST_SAMPLES = 10

    # initialize params
    # NORMALIZER
    normalizer_params = {}
    normalizer_params["class"] = GASSimpleGaussian
    # MODEL
    """ CONCAT MODEL
    # input dim is the flattened shape of the time series, i.e. TS_LEN * n_feat
    model_input_dim = TS_LEN * ts.shape[1]
    HIDDEN_DIM_1 = 20
    HIDDEN_DIM_2 = 20
    ACT = "relu"
    MODEL_OUT_DIM = 20
    model_params = {
        "inp_dim": model_input_dim,
        "hid_dim_1": HIDDEN_DIM_1,
        "hid_dim_2": HIDDEN_DIM_2,
        "activation": ACT,
        "output_dim": MODEL_OUT_DIM,
    }
    model_params["class"] = FFNN
    """
    """ SUM DENORMALIZER """
    # input dim is the flattened shape of the time series, i.e. TS_LEN * n_feat
    model_input_dim = TS_LEN * ts.shape[1]
    HIDDEN_DIM_1 = 20
    HIDDEN_DIM_2 = 20
    ACT = "relu"
    # output dim is LEN_TO_PRED * n_feat
    model_ouput_dim = LEN_TO_PRED * ts.shape[1]
    model_params = {
        "inp_dim": model_input_dim,
        "hid_dim_1": HIDDEN_DIM_1,
        "hid_dim_2": HIDDEN_DIM_2,
        "activation": ACT,
        "output_dim": model_ouput_dim,
    }
    model_params["class"] = FFNN
    # DENORMALIZER
    """ CONCAT DENORMALIZER
    # input dim is output of the model + information about means and vars (i.e. TS_LEN * n_feat for each of the two)
    denorm_input_dim = MODEL_OUT_DIM + 2 * TS_LEN * ts.shape[1]
    # output dim is LEN_TO_PRED * n_feat
    denorm_ouput_dim = LEN_TO_PRED * ts.shape[1]
    denormalizer_params = {
        "input_dim": denorm_input_dim,
        "output_dim": denorm_ouput_dim,
    }
    denormalizer_params["class"] = ConcatDenormalizer
    """
    """ SUM DENORMALIZER """
    # input dim for the mus processer is ts_len * n_feat
    # same for the vars processer
    denorm_input_dim = TS_LEN * ts.shape[1]
    # output dim is LEN_TO_PRED * n_feat
    denorm_ouput_dim = LEN_TO_PRED * ts.shape[1]
    denormalizer_params = {
        "input_dim": denorm_input_dim,
        "output_dim": denorm_ouput_dim,
    }
    denormalizer_params["class"] = SumDenormalizer

    # training params
    CRITERION_NAME = "mse"
    OPTIMIZER_NAME = "adam"
    OPTIMIZER_PARAMS = {}
    BATCH_SIZE = 32
    NUM_EPOCHS = 3

    run_experiment(
        ts,
        PERC_TRAINING_TS,
        TS_LEN,
        LEN_TO_PRED,
        N_TRAIN_SAMPLES,
        N_TEST_SAMPLES,
        normalizer_params,
        model_params,
        denormalizer_params,
        CRITERION_NAME,
        OPTIMIZER_NAME,
        OPTIMIZER_PARAMS,
        BATCH_SIZE,
        NUM_EPOCHS,
    )
