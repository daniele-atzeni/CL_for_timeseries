from default_parameters import *

from run_experiment.gas_experiment import run_gas_experiment


if __name__ == "__main__":
    DATASET_TYPE = "gluonts"  # "synthetic"
    DATASET_NAME = "fred_md"  # gluonts names/custom_name
    # DATASET_NAME = "tourism_monthly"  # gluonts names/custom_name
    DATASET_PARAMS = real_world_data_params  # synthetic_generation_params
    DATASET_PARAMS["multivariate"] = False  # or True
    DATASET_PARAMS["prediction_length"] = None
    DATASET_FILE_FOLDER = None  # "tsf_data"
    # if None, dataset is obtained from GluonTS, if str from file

    NORMALIZER_NAME = "gas_t_student"  # "gas_simple_gaussian", "gas_complex_gaussian"
    NORMALIZER_INITIAL_GUESSES = gas_t_stud_initial_guesses  # gas_{name}_*
    NORMALIZER_BOUNDS = gas_t_stud_bounds
    NORMALIZER_PARAMS = gas_t_stud_params

    MEAN_LAYER_NAME = "gas"  # "gas" or "linear"
    MEAN_LAYER_PARAMS = (
        gas_mean_layer_params if MEAN_LAYER_NAME == "gas" else linear_mean_layer_params
    )

    DL_MODEL_LIBRARY = "gluonts"  # "gluonts" or "torch"
    # "feedforward" or "transformer" or "deepar" or "wavenet" or "mqcnn"
    DL_MODEL_NAME = "deepar"

    if DL_MODEL_NAME == "transformer":
        DL_MODEL_PARAMS = gluonts_transformer_params
    elif DL_MODEL_NAME == "deepar":
        DL_MODEL_PARAMS = gluonts_deepar_params
    elif DL_MODEL_NAME == "wavenet":
        DL_MODEL_PARAMS = gluonts_wavenet_params
    elif DL_MODEL_NAME == "mqcnn":
        DL_MODEL_PARAMS = gluonts_mqcnn_params
    else:
        DL_MODEL_PARAMS = gluonts_feedforward_params

    PROBABILISTIC = True
    N_TRAINING_SAMPLES = 5000
    N_TEST_SAMPLES = 1000

    ROOT_FOLDER = f"ZZZZZTEST_{DATASET_NAME}_{NORMALIZER_NAME}_{MEAN_LAYER_NAME}_{DL_MODEL_LIBRARY}_{DL_MODEL_NAME}"
    if DATASET_PARAMS["multivariate"]:
        ROOT_FOLDER += "_multivariate"

    run_gas_experiment(
        DATASET_NAME,
        DATASET_TYPE,
        DATASET_PARAMS,
        ROOT_FOLDER,
        NORMALIZER_NAME,
        NORMALIZER_INITIAL_GUESSES,
        NORMALIZER_BOUNDS,
        MEAN_LAYER_NAME,
        DL_MODEL_LIBRARY,
        DL_MODEL_NAME,
        DATASET_FILE_FOLDER,
        normalizer_params=NORMALIZER_PARAMS,
        mean_layer_params=MEAN_LAYER_PARAMS,
        dl_model_params=DL_MODEL_PARAMS,
        n_training_samples=N_TRAINING_SAMPLES,
        n_test_samples=N_TEST_SAMPLES,
        probabilistic=PROBABILISTIC,
    )
