import torch

from normalizer import GASSimpleGaussian
from models import FFNN
from denormalizer import SumDenormalizer

from experiment_old import run_torch_experiment


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

    run_torch_experiment(
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
