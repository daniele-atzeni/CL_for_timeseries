import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
from gluonts.evaluation import make_evaluation_predictions
import argparse
import pickle

from run_nonorm_w_tuning import Objective as nonorm_obj
from run_norm_w_tuning import Objective as norm_obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--model_choice', type=str)
    parser.add_argument('--mean_str', type=float)
    parser.add_argument('--var_str', type=float)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    model_choice = args.model_choice
    mean_str = args.mean_str
    var_str = args.var_str
    # model_choice = 'deepar'
    # dataset_name = 'nn5_weekly'
    # mean_str = 0.001
    # var_str = 0.001
    nonorm_params_path = f'saved_nonorm_{model_choice}_{dataset_name}_False/params.json'
    norm_params_path = f'saved_GAS_{model_choice}_{dataset_name}_mean_str_{mean_str}_var_str_{var_str}/params.json'

    NORMALIZER_NAME = "gas_t_student"
    MEAN_MODEL = "gas"
    DL_MODEL = model_choice
    DATASET_NAME = dataset_name
    DL_MODEL_LIBRARY = "gluonts"
    DATASET_TYPE = "gluonts"    # gluonts or synthetic
    multivariate = ""
    ROOT_FOLDER = f"RESULTS_{DATASET_NAME}_{DL_MODEL}_{NORMALIZER_NAME}_{MEAN_MODEL}_{DL_MODEL_LIBRARY}_{mean_str}_{var_str}" + multivariate


    ####### Load Means and Vars #######
    normalizer_folder = os.path.join(ROOT_FOLDER, "normalizer")
    init_params_norm_filename = os.path.join(normalizer_folder, 'init_params.json')
    ###
    means_folder = os.path.join(normalizer_folder, 'means')
    train_means_folder = os.path.join(means_folder, 'train')
    test_means_folder = os.path.join(means_folder, 'test')
    vars_folder = os.path.join(normalizer_folder, 'vars')
    train_vars_folder = os.path.join(vars_folder, 'train')
    test_vars_folder = os.path.join(vars_folder, 'test')
    norm_ts_folder = os.path.join(normalizer_folder, 'normalized_ts')
    train_norm_ts_folder = os.path.join(norm_ts_folder, 'train')
    test_norm_ts_folder = os.path.join(norm_ts_folder, 'test')
    params_folder = os.path.join(normalizer_folder, 'normalizer_params')
    train_params_folder = os.path.join(params_folder, 'train')
    test_params_folder = os.path.join(params_folder, 'test')

    train_means = []
    for i in range(len(os.listdir(train_means_folder))):
        with open(os.path.join(train_means_folder, f'ts_{i}.pkl'), 'rb') as f:
            train_means.append(pickle.load(f))
    test_means = []
    for i in range(len(os.listdir(test_means_folder))):
        with open(os.path.join(test_means_folder, f'ts_{i}.pkl'), 'rb') as f:
            test_means.append(pickle.load(f))
    train_vars = []
    for i in range(len(os.listdir(train_vars_folder))):
        with open(os.path.join(train_vars_folder, f'ts_{i}.pkl'), 'rb') as f:
            train_vars.append(pickle.load(f))
    test_vars = []
    for i in range(len(os.listdir(test_vars_folder))):
        with open(os.path.join(test_vars_folder, f'ts_{i}.pkl'), 'rb') as f:
            test_vars.append(pickle.load(f))
    train_params = []
    for i in range(len(os.listdir(train_params_folder))):
        with open(os.path.join(train_params_folder, f'ts_{i}.pkl'), 'rb') as f:
            train_params.append(pickle.load(f))
    test_params = []
    for i in range(len(os.listdir(test_params_folder))):
        with open(os.path.join(test_params_folder, f'ts_{i}.pkl'), 'rb') as f:
            test_params.append(pickle.load(f))
    test_norm_ts = []
    for i in range(len(os.listdir(test_norm_ts_folder))):
        with open(os.path.join(test_norm_ts_folder, f'ts_{i}.pkl'), 'rb') as f:
            test_norm_ts.append(pickle.load(f))
    ##################################


    ####### Train Models with Loaded Params #######
    with open(nonorm_params_path, 'r') as f:
        nonorm_params = json.load(f)
    with open(norm_params_path, 'r') as f:
        norm_params = json.load(f)
    # nonorm_params['trainer:epochs'] = 2
    # norm_params['trainer:epochs'] = 2

    nonorm_obj_instance = nonorm_obj(
        DL_MODEL, DATASET_NAME, 'gpu', None, multivariate=False
    )
    nonorm_res, nonorm_predictor, nonorm_dir_name, nonorm_history = nonorm_obj_instance.train_and_test(nonorm_params, save=True)

    norm_obj_instance = norm_obj(
        DL_MODEL, DATASET_NAME, 'gpu', None, multivariate=False
    )
    norm_res, norm_predictor, norm_dir_name, norm_history = norm_obj_instance.train_and_test(norm_params, save=True)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=nonorm_obj_instance.test,  # validation dataset
        predictor=nonorm_predictor,  # predictor
        num_samples=100,  # number of sample paths we want for evaluation
    )
    forecasts = list(forecast_it)
    nonorm_forecasts = []
    for f in forecasts:
        nonorm_forecasts.append(f.median)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=norm_obj_instance.test,  # validation dataset
        predictor=norm_predictor,  # predictor
        num_samples=100,  # number of sample paths we want for evaluation
    )
    forecasts = list(forecast_it)
    norm_forecasts = []
    for f in forecasts:
        norm_forecasts.append(f.median)

    ##############################################
        

    ####### Plotting #######
        
    import math
    import matplotlib.pyplot as plt

    dir_name = f'comparisons/comparison_{DL_MODEL}_{DATASET_NAME}_mean_str_{mean_str}_var_str_{var_str}'
    os.makedirs(dir_name, exist_ok=True)

    plt.clf()
    plt.plot(nonorm_history.loss_history, label='Training Loss')
    plt.plot(nonorm_history.validation_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.grid(True)
    plt.savefig(f'{dir_name}/learning_curve_no_norm.png')
    plt.show()
    plt.clf()

    plt.plot(norm_history.loss_history, label='Training Loss')
    plt.plot(norm_history.validation_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.grid(True)
    plt.savefig(f'{dir_name}/learning_curve_GAS.png')
    plt.show()
    plt.clf()

    # Determine the layout of the subplots
    num_plots = len(nonorm_obj_instance.test)
    num_cols = 3  # adjust as needed
    num_rows = math.ceil(num_plots / num_cols)

    # Create a figure for the subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 3))
    training_length_to_plot = 10 # training_data.shape[0]

    for item_id, ts in enumerate(nonorm_obj_instance.test):
        # Determine the current subplot
        ax = axs[item_id // num_cols, item_id % num_cols]

        training_data = ts["target"].T[:-nonorm_obj_instance.prediction_length]
        ground_truth = ts["target"].T[-nonorm_obj_instance.prediction_length:]
        nonorm_pred = nonorm_forecasts[item_id]
        norm_pred = norm_forecasts[item_id]
        
        means = test_means[item_id][-nonorm_obj_instance.prediction_length:]
        vars = test_vars[item_id][-nonorm_obj_instance.prediction_length:]
        # norm_ts = test_norm_ts[item_id]

        # Plot the data on the current subplot
        ax.plot(np.arange(training_length_to_plot), training_data[-training_length_to_plot:], label="training data")
        ax.plot(np.arange(training_length_to_plot, training_length_to_plot + ground_truth.shape[0]), ground_truth, label="ground truth")
        ax.plot(np.arange(training_length_to_plot, training_length_to_plot + nonorm_pred.shape[0]), nonorm_pred, label="no norm prediction")
        ax.plot(np.arange(training_length_to_plot, training_length_to_plot + norm_pred.shape[0]), norm_pred, label="GAS prediction")
        
        ax.plot(np.arange(training_length_to_plot, training_length_to_plot + means.shape[0]), means, label="Means")
        ax.plot(np.arange(training_length_to_plot, training_length_to_plot + vars.shape[0]), vars, label="Vars")
        ax.set_title(f"Item ID: {item_id}")
        ax.legend()

    # Show the figure with all subplots
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{dir_name}/pred_plots.png')