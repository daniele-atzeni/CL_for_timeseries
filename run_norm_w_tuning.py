import os
import pickle

import numpy as np

from utils import get_dataset_from_file
from run_experiment.gas_experiment import run_gas_experiment
from default_parameters import *

from gluonts.mx.distribution import StudentTOutput
from gluonts.evaluation import make_evaluation_predictions
from gluonts.mx.trainer import Trainer
from sklearn.metrics import mean_absolute_error
import mxnet as mx
from my_models.gluonts_models.feedforward_linear_means._estimator import (
    SimpleFeedForwardEstimator as FF_gluonts,
)
from my_models.gluonts_models.multivariate_feedforward_linear_means._estimator import (
    SimpleFeedForwardEstimator as FF_gluonts_multivariate,
)

import time
import optuna
import argparse

class Objective:

    def __init__( self, MODEL, DATASET_NAME, ROOT_FOLDER, prediction_length, context_length, ctx, multivariate=False):
        self.train, self.test, self.freq, self.seasonality = get_dataset_from_file(f'{ROOT_FOLDER}/tsf_data/{DATASET_NAME}',prediction_length,context_length)
        self.model = MODEL
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.multivariate = multivariate
        self.ctx = ctx


        DATASET_TYPE = "gluonts"  # "synthetic"
        DATASET_PARAMS = real_world_data_params  # synthetic_generation_params
        DATASET_PARAMS["multivariate"] = self.multivariate  

        NORMALIZER_NAME = "gas_t_student"  # "gas_simple_gaussian", "gas_complex_gaussian"
        NORMALIZER_INITIAL_GUESSES = gas_t_stud_initial_guesses  # gas_{name}_*
        NORMALIZER_BOUNDS = gas_t_stud_bounds
        NORMALIZER_PARAMS = gas_t_stud_params

        MEAN_LAYER_NAME = "linear"  # TODO: gas
        MEAN_LAYER_PARAMS = linear_mean_layer_params

        DL_MODEL_LIBRARY = "gluonts"  # "torch"
        DL_MODEL_NAME = "feedforward"  # TODO: "transformer"
        DL_MODEL_PARAMS = gluonts_multivariate_feedforward_params

        N_TRAINING_SAMPLES = 5000
        N_TEST_SAMPLES = 1000

        ROOT_FOLDER = (
            f"RESULTS_{DATASET_NAME}_{NORMALIZER_NAME}_{MEAN_LAYER_NAME}_{DL_MODEL_LIBRARY}"
        )
        if DATASET_PARAMS["multivariate"]:
            ROOT_FOLDER += "_multivariate"

        # run for the first time to get all the means and stuff
        training_params = run_gas_experiment(
          DATASET_NAME,
          prediction_length,
          context_length,
          DATASET_TYPE,
          DATASET_PARAMS,
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
        )

        # get the parameters needed to run the model part

        self.n_features, self.context_length, self.prediction_length, self.dataset, self.weights, self.bias, self.dl_model_name, self.dl_model_params, self.folders = training_params

        self.train, self.test = self.dataset

    def get_params(self, trial) -> dict:

        if self.model == 'ffn':
          return {
              "num_hidden_dimensions": [trial.suggest_int("hidden_dim_{}".format(i), 10, 100) for i in range(trial.suggest_int("num_layers", 1, 5))],
              "trainer:learning_rate": trial.suggest_loguniform("trainer:learning_rate", 1e-5, 1e-1),
              "trainer:epochs": trial.suggest_int("trainer:epochs", 10, 100),
          }
        elif self.model == 'transformer':
          # num_heads must divide model_dim
          valid_pairs = [ (i,d) for i in range(10,101) for d in range(1,11) if i%d == 0  ]
          model_dim_num_heads_pair = trial.suggest_categorical("model_dim_num_heads_pair", valid_pairs)

          return {
              "inner_ff_dim_scale": trial.suggest_int("inner_ff_dim_scale", 1, 5),
              "model_dim": model_dim_num_heads_pair[0],
              "embedding_dimension": trial.suggest_int("embedding_dimension", 1, 10),
              "num_heads": model_dim_num_heads_pair[1],
              "dropout_rate": trial.suggest_uniform("dropout_rate", 0.0, 0.5),
              "trainer:learning_rate": trial.suggest_loguniform("trainer:learning_rate", 1e-5, 1e-1),
              "trainer:epochs": trial.suggest_int("trainer:epochs", 10, 100),
          }


    def __call__(self, trial):

        params = self.get_params(trial)

        mean_layer = mx.gluon.nn.HybridSequential()
        mean_layer.add(
            mx.gluon.nn.Dense(
                units= self.prediction_length * self.n_features,
                weight_initializer=mx.init.Constant(self.weights),
                bias_initializer=mx.init.Constant(self.bias),  # type: ignore # bias is a numpy array, don't know the reasons for this typing error
            )
        )
        mean_layer.add(
            mx.gluon.nn.HybridLambda(
                lambda F, o: F.reshape(
                    o, (-1, self.prediction_length * self.n_features)
                )  # no need for that but just to be sure
            )
        )
        # freeze the parameters
        for param in mean_layer.collect_params().values():
            param.grad_req = "null"

        if self.model == 'ffn':
          estimator = FF_gluonts(
                mean_layer,
                StudentTOutput(),
                prediction_length=self.prediction_length,
                context_length=self.context_length,
                num_hidden_dimensions= params['num_hidden_dimensions'], #num_hidden_dimensions,
                trainer=Trainer(ctx=self.ctx,epochs=params['trainer:epochs'], learning_rate=params['trainer:learning_rate'], num_batches_per_epoch=100),
            )
        elif self.model == 'transformer':
          estimator = TransformerEstimator(
              freq=self.freq,
              context_length=self.context_length,
              prediction_length=self.prediction_length,
              inner_ff_dim_scale= params['inner_ff_dim_scale'],
              model_dim= params['model_dim'],
              embedding_dimension= params['embedding_dimension'],
              num_heads= params['num_heads'],
              dropout_rate= params['dropout_rate'],
              trainer=Trainer(ctx=self.ctx,epochs=params['trainer:epochs'], learning_rate=params['trainer:learning_rate'], num_batches_per_epoch=100),
          )

        ## TRAIN
        predictor = estimator.train(self.train)
        ## EVALUATE
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=self.test,  # test dataset
            predictor=predictor,  # predictor
            num_samples=100,  # number of sample paths we want for evaluation
        )

        forecasts = list(forecast_it)

        final_forecasts = []
        for f in forecasts:
            final_forecasts.append(f.median)

        mase_metrics = []
        for item_id, ts in enumerate(self.test):
          training_data = ts["target"].T[:-self.prediction_length]
          ground_truth = ts["target"].T[-self.prediction_length:]

          y_pred_naive = np.array(training_data)[:-int(self.seasonality)]
          mae_naive = mean_absolute_error(np.array(training_data)[int(self.seasonality):], y_pred_naive, multioutput="uniform_average")

          mae_score = mean_absolute_error(
              np.array(ground_truth),
              final_forecasts[item_id],
              sample_weight=None,
              multioutput="uniform_average",
          )

          epsilon = np.finfo(np.float64).eps
          if mae_naive == 0:
            continue
          mase_score = mae_score / np.maximum(mae_naive, epsilon)


          mase_metrics.append(mase_score)

        return np.mean(mase_metrics)
    

def run(DATASET_NAME, ROOT_FOLDER, model_choice, prediction_length, context_length, ctx, n_trials):

    start_time = time.time()
    study = optuna.create_study(direction="minimize")
    study.optimize(
        Objective(
            model_choice,DATASET_NAME, ROOT_FOLDER, prediction_length, context_length, ctx, multivariate=False # multivar seems 0.02-0.03 better
        ),
        n_trials=n_trials,
    )

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    print(time.time() - start_time)

    file_path = "output.txt"
    with open(file_path, "a") as file:
        file.write(f' ########################### {model_choice} with gas norm on {DATASET_NAME} Final MASE: {trial.value}\n')


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset_name', type=str)
  parser.add_argument('--root_folder', type=str)
  parser.add_argument('--model_choice', type=str)
  parser.add_argument('--prediction_length', type=int)
  parser.add_argument('--context_length', type=int)
  parser.add_argument('--ctx', type=str)
  parser.add_argument('--n_trials', type=int, default=20)
  # parser.add_argument('--use_tsf', action='store_true')
  args = parser.parse_args()
  print(args)
  run(args.dataset_name, args.root_folder, args.model_choice, args.prediction_length, args.context_length , args.ctx, args.n_trials)