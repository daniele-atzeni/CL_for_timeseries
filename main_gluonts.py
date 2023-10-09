from normalizer import GASSimpleGaussian
from models import FFNN
from denormalizer import SumDenormalizer

from experiment import run_experiment

from gluonts.dataset.repository import get_dataset
from gluonts.mx import SimpleFeedForwardEstimator, Trainer


from utils import convert_gluon_dataset_to_train_tensor


if __name__ == "__main__":
    # get the dataset
    dataset_name = "m4_hourly"
    gluon_ts = get_dataset(dataset_name)
    if gluon_ts is None:
        raise ValueError(f"Dataset {dataset_name} not found")
    if gluon_ts.test is None:
        raise ValueError(f"Dataset {dataset_name} has no test set")
    if gluon_ts.metadata.prediction_length is None:
        raise ValueError(f"Dataset {dataset_name} has no prediction length")

    print(f"Dataset {dataset_name} loaded successfully")
    print(
        f"Dataset {dataset_name} has prediction length {gluon_ts.metadata.prediction_length}"
    )

    ts = convert_gluon_dataset_to_train_tensor(gluon_ts.test)

    print(
        f"Dataset {dataset_name} converted to tensor with shape {ts.shape} successfully"
    )

    # dataset params
    PERC_TRAINING_TS = 0.75
    ts_len = gluon_ts.metadata.prediction_length
    len_to_pred = gluon_ts.metadata.prediction_length
    N_TRAIN_SAMPLES = 20
    N_TEST_SAMPLES = 10

    # initialize params
    # NORMALIZER
    normalizer_params = {}
    normalizer_params["class"] = GASSimpleGaussian
    # MODEL
    trainer = Trainer(
        ctx="cpu",  # type: ignore      don't know why he doesn't like this
        epochs=5,
        learning_rate=1e-3,
        hybridize=False,
        num_batches_per_epoch=100,
    )
    model_params = {
        "num_hidden_dimensions": [10],
        "prediction_length": gluon_ts.metadata.prediction_length,
        "context_length": 2 * gluon_ts.metadata.prediction_length,
        "trainer": trainer,
    }

    model_params["class"] = SimpleFeedForwardEstimator
    # DENORMALIZER
    # input dim for the mus processer is ts_len * n_feat
    # same for the vars processer
    denorm_input_dim = ts_len * ts.shape[1]
    # output dim is LEN_TO_PRED * n_feat
    denorm_ouput_dim = len_to_pred * ts.shape[1]
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
        ts_len,
        len_to_pred,
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
