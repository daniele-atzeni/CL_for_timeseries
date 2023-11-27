import os

from default_parameters import *

from run_experiment.gas_experiment import run_gas_experiment


if __name__ == "__main__":
    DATASET_TYPE = "synthetic"  # "gluonts"
    DATASET_NAME = "seasonal"  # custom names/gluonts names
    DATASET_PARAMS = synthetic_generation_params  # real_world_data_params
    DATASET_PARAMS["multivariate"] = True  # or False

    NORMALIZER_NAME = "gas_t_student"  # "gas_simple_gaussian", "gas_complex_gaussian"
    NORMALIZER_INITIAL_GUESSES = gas_t_stud_initial_guesses  # gas_{name}_*
    NORMALIZER_BOUNDS = gas_t_stud_bounds
    NORMALIZER_PARAMS = gas_t_stud_params

    MEAN_LAYER_NAME = "linear"
    MEAN_LAYER_PARAMS = linear_mean_layer_params

    DL_MODEL_LIBRARY = "gluonts"  # "torch"
    DL_MODEL_NAME = "multivariate_feedforward"  # "feedforward"
    DL_MODEL_PARAMS = gluonts_multivariate_feedforward_params

    N_TRAINING_SAMPLES = 5000
    N_TEST_SAMPLES = 1000

    ROOT_FOLDER = (
        f"RESULTS_STRENGTHS_{DATASET_NAME}_{NORMALIZER_NAME}_{DL_MODEL_LIBRARY}"
    )
    if DATASET_PARAMS["multivariate"]:
        ROOT_FOLDER += "_multivariate"

    strengths = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.49, 0.499, 0.5]
    for strength in strengths:
        print(40 * "=")
        print(f"Trying strength {strength}")
        STRENGTH_FOLDER = os.path.join(ROOT_FOLDER, f"STRENGTH_{strength}")

        NORMALIZER_PARAMS["mean_strength"] = strength

        run_gas_experiment(
            DATASET_NAME,
            DATASET_TYPE,
            DATASET_PARAMS,
            STRENGTH_FOLDER,
            NORMALIZER_NAME,
            NORMALIZER_INITIAL_GUESSES,
            NORMALIZER_BOUNDS,
            MEAN_LAYER_NAME,
            DL_MODEL_LIBRARY,
            DL_MODEL_NAME,
            NORMALIZER_PARAMS,
            MEAN_LAYER_PARAMS,
            DL_MODEL_PARAMS,
            N_TRAINING_SAMPLES,
            N_TEST_SAMPLES,
            stop_after_normalizer=True,
        )
