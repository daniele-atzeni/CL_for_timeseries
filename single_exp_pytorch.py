from experiments.gas_experiment import run_gas_experiment

if __name__ == "__main__":
    DATASET_NAME = "nn5_weekly"
    DATASET_TYPE = "gluonts"

    ROOT_FOLDER = "UNIVARIATE_TORCH_RESULTS_GAS"

    NORMALIZER_NAME = "gas_complex_gaussian"
    NORMALIZER_PARAMS = {"eps": 1e-9, "regularization": "full"}

    MEAN_LAYER_NAME = "linear"
    MEAN_LAYER_PARAMS = {}

    DL_MODEL_LIBRARY = "torch"
    DL_MODEL_NAME = "feedforward"
    DL_MODEL_PARAMS = {
        "main_model": {
            "num_hidden_dimensions": [128, 17],
        },
        "training": {
            "loss": "mse",
            "epochs": 5,
            "optimizer": "adam",
            "learning_rate": 1e-3,
            "batch_size": 128,
        },
        "prediction": {},
        "evaluation": {},
    }

    N_TRAINING_SAMPLES = 5000
    N_TEST_SAMPLES = 1000

    run_gas_experiment(
        DATASET_NAME,
        DATASET_TYPE,
        ROOT_FOLDER,
        NORMALIZER_NAME,
        MEAN_LAYER_NAME,
        DL_MODEL_LIBRARY,
        DL_MODEL_NAME,
        NORMALIZER_PARAMS,
        MEAN_LAYER_PARAMS,
        DL_MODEL_PARAMS,
        N_TRAINING_SAMPLES,
        N_TEST_SAMPLES,
    )
