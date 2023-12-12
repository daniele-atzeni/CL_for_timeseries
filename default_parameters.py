import numpy as np

small_val = 1e-6

# DATASET GENERATION PARAMS
synthetic_generation_params = {
    "train_length": 100,
    "prediction_length": 10,
    "num_timeseries": 10,
}
real_world_data_params = {}

# NORMALIZER INITIAL GUESSES
# simple gauss
gas_simple_gauss_initial_guesses = np.array([0.001, 0.001, 0, 1], dtype="float")
gas_simple_gauss_bounds = (
    (None, None),
    (0 + small_val, 1 - small_val),
    (0 + small_val, 1 - small_val),
    (0 + small_val, 1 - small_val),
)
gas_simple_gauss_params = {"eps": 1e-9}

# complex gauss
gas_complex_gauss_initial_guesses = np.array([0.001, 0.001, 0, 1], dtype="float")
gas_complex_gauss_bounds = (
    (None, None),
    (0 + small_val, 1 - small_val),
    (0 + small_val, 1 - small_val),
    (0 + small_val, 1 - small_val),
)
gas_complex_gauss_params = {"eps": 1e-9, "regularization": "full"}

# student t
gas_t_stud_initial_guesses = np.array(
    [0, 1, 0 + small_val, 0 + small_val, 1, 1, 0, 0, 3], dtype=np.float32
)
gas_t_stud_bounds = (
    (None, None),
    (0, None),
    (0 + small_val, 1 - small_val),
    (0 + small_val, 1 - small_val),
    (0 + small_val, 1 - small_val),
    (0 + small_val, 1 - small_val),
    (None, None),
    (0 + small_val, None),
    (2 + small_val, 50),
)
gas_t_stud_params = {
    "mean_strength": 0.5,
    "var_strength": 0.5,
    "eps": 1e-6,
    # "max_var": 1000,
}

# MEAN LAYER PARAMS
# linear
linear_mean_layer_params = {}

# DL MODEL PARAMS
# gluonts
gluonts_multivariate_feedforward_params = {
    "main_model": {
        "num_hidden_dimensions": [512, 256],
    },
    "training": {
        "epochs": 5,
        "learning_rate": 5 * 1e-4,
        "num_batches_per_epoch": 100,
    },
    "prediction": {"num_samples": 100},
    "evaluation": {"quantiles": [0.1, 0.5, 0.9]},
}

# torch
torch_feedforward_params = {
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
