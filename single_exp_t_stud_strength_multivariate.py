import numpy as np

from run_experiment.gas_experiment import run_gas_experiment


if __name__ == "__main__":
    DATASET_NAME = "nn5_daily_without_missing"
    DATASET_TYPE = "gluonts"

    # this parameters are actually useless
    MEAN_LAYER_NAME = "linear"
    MEAN_LAYER_PARAMS = {}

    DL_MODEL_LIBRARY = "gluonts"
    DL_MODEL_NAME = "multivariate_feedforward"

    DL_MODEL_PARAMS = {
        "main_model": {
            "num_hidden_dimensions": [128, 17],
        },
        "training": {
            "epochs": 5,
            "learning_rate": 1e-3,
            "num_batches_per_epoch": 100,
        },
        "prediction": {"num_samples": 100},
        "evaluation": {"quantiles": [0.1, 0.5, 0.9]},
    }

    N_TRAINING_SAMPLES = 5000
    N_TEST_SAMPLES = 1000
    #

    for mean_strength in [0.1, 0.2, 0.3, 0.4, 0.5]:
        print(f"Trying strength {mean_strength}")

        ROOT_FOLDER = f"T_STUD_CHANGING_STRENGTH_MULTIVARIATE_RESULTS_{mean_strength}"

        NORMALIZER_NAME = "gas_t_student"
        NORMALIZER_INITIAL_GUESSES = np.array([0.001, 0.001, 0, 1, 3], dtype="float")
        NORMALIZER_BOUNDS = (
            (None, None),
            (0.00001, 1),
            (0, 0.999),
            (0, 0.999),
            (2.00001, 50),
        )
        NORMALIZER_PARAMS = {
            "mean_strength": mean_strength,
            "var_strength": 0.5,
            "eps": 1e-9,
        }

        run_gas_experiment(
            DATASET_NAME,
            DATASET_TYPE,
            ROOT_FOLDER,
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
            multivariate=True,
            stop_after_normalizer=True,
        )