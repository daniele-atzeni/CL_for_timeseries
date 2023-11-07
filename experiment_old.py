import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import MSELoss

from gluonts.dataset import Dataset

from training import train
from utils_old import create_forecasting_dataset, convert_gluon_dataset_to_train_tensor


def run_torch_experiment(
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

    train(
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


def run_gluonts_experiment(
    ts: Dataset,
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
    ts_tensor = convert_gluon_dataset_to_train_tensor(ts.test)  # type: ignore      we know that dataset has test attr
    # ----------- DATASET PREPARATION --------

    # dividing the time series into training and test. How?
    # Until a certain time is train, then is test
    end_training_ind = int(perc_training_ts * ts_tensor.shape[0])
    train_ts = ts_tensor[:end_training_ind]
    test_ts = ts_tensor[end_training_ind:]

    # create the dataset
    # train_dataset = create_forecasting_dataset(train_norm_ts, train_mus, train_vars, TS_LEN, LEN_TO_PRED, N_SAMPLES)
    # test_dataset = create_forecasting_dataset(test_norm_ts, test_mus, test_vars, TS_LEN, LEN_TO_PRED, N_SAMPLES)
    train_dataset = create_forecasting_dataset(
        train_ts, ts_len, len_to_pred, n_train_sample
    )
    print("Tensor train Datasets created")
    test_dataset = create_forecasting_dataset(
        test_ts, ts_len, len_to_pred, n_test_sample
    )
    print("Tensor test Datasets created")
    # ----------------------------------------

    # ----------- INIT NORMALIZER ------------
    normalizer_class = normalizer_params.pop("class")
    normalizer = normalizer_class(**normalizer_params)
    # WARM UP PHASE: for GAS, compute static parameters and precompute means and vars of all the ts
    normalizer.warm_up(ts_tensor, train_ts)
    # ----------------------------------------

    # ----------- INIT MODEL -----------------
    model_class = model_params.pop("class")
    model = model_class(**model_params)
    # ----------------------------------------

    # ----------- INIT DENORMALIZER ----------
    denorm_class = denormalizer_params.pop("class")
    denormalizer = denorm_class(**denormalizer_params)
    # ----------------------------------------

    print("Initialized normalizer, ts_model and denormalizer")

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

    print("Everything initialized, starting training")

    train(
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


def run_experiment(
    ts: torch.Tensor | Dataset,
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
    if isinstance(ts, Tensor):
        return run_torch_experiment(
            ts,
            perc_training_ts,
            ts_len,
            len_to_pred,
            n_train_sample,
            n_test_sample,
            normalizer_params,
            model_params,
            denormalizer_params,
            criterion_name,
            optimizer_name,
            optimizer_params,
            batch_size,
            num_epochs,
        )
    elif isinstance(ts, Dataset):
        return run_gluonts_experiment(
            ts,
            perc_training_ts,
            ts_len,
            len_to_pred,
            n_train_sample,
            n_test_sample,
            normalizer_params,
            model_params,
            denormalizer_params,
            criterion_name,
            optimizer_name,
            optimizer_params,
            batch_size,
            num_epochs,
        )
    else:
        raise NotImplementedError(f"Type {type(ts)} not implemented")
