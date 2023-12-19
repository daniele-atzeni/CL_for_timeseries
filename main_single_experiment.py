from default_parameters import *

from run_experiment.gas_experiment import run_gas_experiment


if __name__ == "__main__":
    DATASET_TYPE = "gluonts"  # "synthetic"
    DATASET_NAME = "nn5_weekly"  # gluonts names/custom_name
    DATASET_PARAMS = real_world_data_params  # synthetic_generation_params
    DATASET_PARAMS["multivariate"] = True  # or False
    DATASET_FILE_FOLDER = None  # "tsf_data"
    # if None, dataset is obtained from GluonTS, if str from file

    NORMALIZER_NAME = "gas_t_student"  # "gas_simple_gaussian", "gas_complex_gaussian"
    NORMALIZER_INITIAL_GUESSES = gas_t_stud_initial_guesses  # gas_{name}_*
    NORMALIZER_BOUNDS = gas_t_stud_bounds
    NORMALIZER_PARAMS = gas_t_stud_params

    MEAN_LAYER_NAME = "gas"  # TODO: gas
    MEAN_LAYER_PARAMS = (
        gas_mean_layer_params if MEAN_LAYER_NAME == "gas" else linear_mean_layer_params
    )

    DL_MODEL_LIBRARY = "gluonts"  # "torch"
    DL_MODEL_NAME = "feedforward"  # "feedforward" TODO: transformer
    DL_MODEL_PARAMS = (
        gluonts_transformer_params
        if DL_MODEL_NAME == "transformer"
        else gluonts_feedforward_params
    )

    N_TRAINING_SAMPLES = 5000
    N_TEST_SAMPLES = 1000

    ROOT_FOLDER = (
        f"RESULTS_{DATASET_NAME}_{NORMALIZER_NAME}_{MEAN_LAYER_NAME}_{DL_MODEL_LIBRARY}"
    )
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
        NORMALIZER_PARAMS,
        MEAN_LAYER_PARAMS,
        DL_MODEL_PARAMS,
        N_TRAINING_SAMPLES,
        N_TEST_SAMPLES,
    )
