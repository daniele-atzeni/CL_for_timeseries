import os
import pickle
import warnings
warnings.filterwarnings("ignore")
import numpy as np

from utils import init_folder, load_list_of_elements, get_dataset_and_metadata, get_dataset_from_file

import time
import optuna
import argparse


from gluonts.evaluation import make_evaluation_predictions
from gluonts.mx.trainer import Trainer
from gluonts.mx.trainer.callback import TrainingHistory
from gluonts.mx.distribution import StudentTOutput, MultivariateGaussianOutput
from sklearn.metrics import mean_absolute_error
from gluonts.dataset.multivariate_grouper import MultivariateGrouper

from gluonts.mx.model.simple_feedforward import SimpleFeedForwardEstimator
from my_models.gluonts_models.univariate.feedforward_point import SimpleFeedForwardEstimator as FF_gluonts_univariate_point
from my_models.gluonts_models.ffn_multivar import SimpleFeedForwardEstimator as FF_gluonts_multivariate
from gluonts.mx.model.transformer import TransformerEstimator
from gluonts.mx.model.deepar import DeepAREstimator 
from gluonts.mx.model.wavenet import WaveNetEstimator
from gluonts.mx.model.seq2seq import MQCNNEstimator

from my_models.gluonts_models.batch_norm.simple_feedforward import SimpleFeedForwardEstimator as FF_batch_norm
from my_models.gluonts_models.batch_norm.seq2seq import MQCNNEstimator as MQCNN_batch_norm
from my_models.gluonts_models.batch_norm.transformer import TransformerEstimator as Transformer_batch_norm

import matplotlib.pyplot as plt
import json
from pathlib import Path
from gluonts.dataset.split import split
from gluonts.dataset.common import ListDataset
import copy

from data_manager import GluonTSDataManager

seasonality = {
    'nn5_weekly': 52.17857142857143,
    'us_births_dataset': 7,
    'weather': 7,
    'sunspot_without_missing': 7,
    'solar_10_minutes': 144,
    'hospital': 12,
    'rideshare_without_missing': 24,
    'fred_md': 12
}

SEASONALITY_MAP = {
   "minutely": [1440, 10080, 525960],
   "10_minutes": [144, 1008, 52596],
   "half_hourly": [48, 336, 17532],
   "H": [24, 168, 8766],
   "D": 7,
   "W": 365.25/7,
   "M": 12,
   "quarterly": 4,
   "Y": 1
}

class Objective:

    def __init__( self, MODEL, DATASET_NAME, ctx, DATASET_FILE_FOLDER, multivariate=False, standardize=False, batch_norm=False, mean_scaling=False):
        
        data_manager = GluonTSDataManager(DATASET_NAME, multivariate, DATASET_FILE_FOLDER, standardize)
        self.data_manager = data_manager
        self.n_features = data_manager.n_features
        self.context_length = data_manager.context_length
        self.prediction_length = data_manager.prediction_length
        self.train_original = data_manager.train_dataset
        self.test = data_manager.test_dataset
        self.freq = data_manager.freq
        self.dataset_name = DATASET_NAME
        self.standardize = standardize
        self.batch_norm = batch_norm
        self.mean_scaling = mean_scaling
        self.seasonality = SEASONALITY_MAP[self.freq]
        if isinstance(self.seasonality, list):
          self.seasonality = min(self.seasonality) # Use to calculate MASE

        self.model = MODEL
        self.multivariate = multivariate
        self.ctx = ctx

        self.train, test_template = split(self.train_original, offset=-self.prediction_length)
        validation = test_template.generate_instances(
            prediction_length=self.prediction_length,
        )
        # Assuming `validation` is a list of (input, output) pairs
        validation_data = [
            {
                "start": v[0]["start"],  # replace with the actual start time
                "target": np.concatenate([v[0]['target'], v[1]['target']]),
            }
            for v in validation
        ]

        self.validation = ListDataset(validation_data, freq=self.freq)

        print(self.model, self.multivariate, self.ctx)


    def get_params(self, trial) -> dict:

        if self.model == 'feedforward':
          return {
              "num_hidden_dimensions": [trial.suggest_int("hidden_dim_{}".format(i), 10, 100) for i in range(trial.suggest_int("num_layers", 1, 5))],
              "trainer:learning_rate": trial.suggest_loguniform("trainer:learning_rate", 1e-6, 1e-3),
              "trainer:epochs": trial.suggest_int("trainer:epochs", 10, 100),
          }
        elif self.model == 'wavenet':
          return {
              "trainer:learning_rate": trial.suggest_loguniform("trainer:learning_rate", 1e-6, 1e-3),
              "trainer:epochs": trial.suggest_int("trainer:epochs", 10, 100),
          }
        elif self.model == 'mqcnn':
          return {
              "trainer:learning_rate": trial.suggest_loguniform("trainer:learning_rate", 1e-6, 1e-3),
              "trainer:epochs": trial.suggest_int("trainer:epochs", 10, 100),
          }
        elif self.model == 'deepar':
           return {
              "num_cells": trial.suggest_int("num_cells", 10, 100),
              "num_layers": trial.suggest_int("num_layers", 1, 10),
              "trainer:learning_rate": trial.suggest_loguniform("trainer:learning_rate", 1e-6, 1e-3),
              "trainer:epochs": trial.suggest_int("trainer:epochs", 10, 100)
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
              "trainer:learning_rate": trial.suggest_loguniform("trainer:learning_rate", 1e-6, 1e-3),
              "trainer:epochs": trial.suggest_int("trainer:epochs", 10, 100),
          }


    def __call__(self, trial):

        params = self.get_params(trial)

        return self.train_and_test(params)

    def train_and_test(self, params, save=False):

      history = TrainingHistory()
      # from my_models.gluonts_models.univariate.probabilistic_forecast.feedforward_test._estimator import SimpleFeedForwardEstimator as FF_test
      if self.model == 'feedforward' and not self.multivariate:
        if self.batch_norm:
          estimator = FF_batch_norm( # SimpleFeedForwardEstimator FF_test
              num_hidden_dimensions= params['num_hidden_dimensions'], #num_hidden_dimensions,
              prediction_length=self.prediction_length,
              context_length=self.context_length,
              batch_normalization=False,
              mean_scaling=False,
              trainer=Trainer(ctx=self.ctx,epochs=params['trainer:epochs'], learning_rate=params['trainer:learning_rate'],
                              num_batches_per_epoch=100, callbacks=[history]),
          )
        elif self.mean_scaling:
          estimator = SimpleFeedForwardEstimator(
              num_hidden_dimensions= params['num_hidden_dimensions'], #num_hidden_dimensions,
              prediction_length=self.prediction_length,
              context_length=self.context_length,
              batch_normalization=False,
              mean_scaling=True,
              trainer=Trainer(ctx=self.ctx,epochs=params['trainer:epochs'], learning_rate=params['trainer:learning_rate'],
                              num_batches_per_epoch=100, callbacks=[history]),
          )
        else:
          estimator = SimpleFeedForwardEstimator(
              num_hidden_dimensions= params['num_hidden_dimensions'], #num_hidden_dimensions,
              prediction_length=self.prediction_length,
              context_length=self.context_length,
              batch_normalization=False,
              mean_scaling=False,
              trainer=Trainer(ctx=self.ctx,epochs=params['trainer:epochs'], learning_rate=params['trainer:learning_rate'],
                              num_batches_per_epoch=100, callbacks=[history]),
          )
      # elif self.model == 'feedforward' and self.multivariate:
      #   estimator = FF_gluonts_multivariate(
      #       num_hidden_dimensions= params['num_hidden_dimensions'], #num_hidden_dimensions,
      #       prediction_length=self.prediction_length,
      #       context_length=self.context_length,
      #       mean_scaling=False,
      #       # batch_normalization=True,
      #       distr_output=MultivariateGaussianOutput(dim=self.n_features),
      #       trainer=Trainer(ctx=self.ctx,epochs=params['trainer:epochs'], learning_rate=params['trainer:learning_rate'],
      #                        num_batches_per_epoch=100, callbacks=[history]),
      #   )
      elif self.model == 'wavenet':
        estimator = WaveNetEstimator(
            freq=self.freq,
            prediction_length=self.prediction_length,
            trainer=Trainer(ctx=self.ctx,epochs=params['trainer:epochs'], learning_rate=params['trainer:learning_rate'],
                             num_batches_per_epoch=100, callbacks=[history], add_default_callbacks=False),
        )
      elif self.model == 'mqcnn':
        if self.batch_norm:
          estimator = MQCNN_batch_norm(
              freq=self.freq,
              prediction_length=self.prediction_length,
              context_length=self.context_length,
              distr_output=StudentTOutput(),
              quantiles=None,
              scaling=False, # default is none, set True to use
              trainer=Trainer(ctx=self.ctx,epochs=params['trainer:epochs'], learning_rate=params['trainer:learning_rate'],
                              num_batches_per_epoch=100, callbacks=[history], hybridize=False),
          )
        elif self.mean_scaling:
          estimator = MQCNNEstimator(
              freq=self.freq,
              prediction_length=self.prediction_length,
              context_length=self.context_length,
              distr_output=StudentTOutput(),
              quantiles=None,
              scaling=True, 
              trainer=Trainer(ctx=self.ctx,epochs=params['trainer:epochs'], learning_rate=params['trainer:learning_rate'],
                              num_batches_per_epoch=100, callbacks=[history], hybridize=False),
          )
        else:
          estimator = MQCNNEstimator(
              freq=self.freq,
              prediction_length=self.prediction_length,
              context_length=self.context_length,
              distr_output=StudentTOutput(),
              quantiles=None,
              scaling=False, 
              trainer=Trainer(ctx=self.ctx,epochs=params['trainer:epochs'], learning_rate=params['trainer:learning_rate'],
                              num_batches_per_epoch=100, callbacks=[history], hybridize=False),
          )
      elif self.model == 'deepar':
        if self.batch_norm:
          estimator = DeepAREstimator(
              freq=self.freq,
              context_length=self.context_length,
              distr_output=StudentTOutput(),
              prediction_length=self.prediction_length,
              # num_cells= params['num_cells'],
              # num_layers= params['num_layers'],
              scaling=False, # True by default
              trainer=Trainer(ctx=self.ctx,epochs=params['trainer:epochs'], learning_rate=params['trainer:learning_rate'],
                               num_batches_per_epoch=100, callbacks=[history]),
          )
        elif self.mean_scaling:
          estimator = DeepAREstimator(
              freq=self.freq,
              context_length=self.context_length,
              distr_output=StudentTOutput(),
              prediction_length=self.prediction_length,
              # num_cells= params['num_cells'],
              # num_layers= params['num_layers'],
              scaling=True, # True by default
              trainer=Trainer(ctx=self.ctx,epochs=params['trainer:epochs'], learning_rate=params['trainer:learning_rate'],
                              num_batches_per_epoch=100, callbacks=[history]),
          )
        else:
          estimator = DeepAREstimator(
              freq=self.freq,
              context_length=self.context_length,
              distr_output=StudentTOutput(),
              prediction_length=self.prediction_length,
              # num_cells= params['num_cells'],
              # num_layers= params['num_layers'],
              scaling=False, # True by default
              trainer=Trainer(ctx=self.ctx,epochs=params['trainer:epochs'], learning_rate=params['trainer:learning_rate'],
                              num_batches_per_epoch=100, callbacks=[history]),
          )
      elif self.model == 'transformer':
        if self.batch_norm:
          estimator = Transformer_batch_norm(
              freq=self.freq,
              context_length=self.context_length,
              prediction_length=self.prediction_length,
              distr_output=StudentTOutput(),
              # inner_ff_dim_scale= params['inner_ff_dim_scale'],
              # model_dim= params['model_dim'],
              # embedding_dimension= params['embedding_dimension'],
              # num_heads= params['num_heads'],
              # dropout_rate= params['dropout_rate'],
              scaling=False, # True by default False
              trainer=Trainer(ctx=self.ctx,epochs=params['trainer:epochs'], learning_rate=params['trainer:learning_rate'],
                               num_batches_per_epoch=100, callbacks=[history]),
          )
        elif self.mean_scaling:
          estimator = TransformerEstimator(
              freq=self.freq,
              context_length=self.context_length,
              prediction_length=self.prediction_length,
              distr_output=StudentTOutput(),
              # inner_ff_dim_scale= params['inner_ff_dim_scale'],
              # model_dim= params['model_dim'],
              # embedding_dimension= params['embedding_dimension'],
              # num_heads= params['num_heads'],
              # dropout_rate= params['dropout_rate'],
              scaling=True, # True by default False
              trainer=Trainer(ctx=self.ctx,epochs=params['trainer:epochs'], learning_rate=params['trainer:learning_rate'],
                              num_batches_per_epoch=100, callbacks=[history]),
          )
        else:
          estimator = TransformerEstimator(
              freq=self.freq,
              context_length=self.context_length,
              prediction_length=self.prediction_length,
              distr_output=StudentTOutput(),
              # inner_ff_dim_scale= params['inner_ff_dim_scale'],
              # model_dim= params['model_dim'],
              # embedding_dimension= params['embedding_dimension'],
              # num_heads= params['num_heads'],
              # dropout_rate= params['dropout_rate'],
              scaling=False, # True by default False
              trainer=Trainer(ctx=self.ctx,epochs=params['trainer:epochs'], learning_rate=params['trainer:learning_rate'],
                              num_batches_per_epoch=100, callbacks=[history]),
              # trainer=Trainer(ctx=self.ctx,epochs=1, learning_rate=5*1e-4,
              #                  num_batches_per_epoch=100, callbacks=[history]),
          )

      ## TRAIN
      predictor = estimator.train(self.train, self.validation)
      ## EVALUATE
      if not save:
         test = copy.deepcopy(self.validation) # deep copy so unstandardize_data doesn't affect the original test set
      else:
         test = copy.deepcopy(self.test)
      forecast_it, ts_it = make_evaluation_predictions(
          dataset=test,  # validation dataset
          predictor=predictor,  # predictor
          num_samples=100,  # number of sample paths we want for evaluation
      )

      forecasts = list(forecast_it)

      final_forecasts = []
      for f in forecasts:
          if self.model == 'mqcnn':
            final_forecasts.append(f.mean) # mqcnn doesnt support median or quantiles atm
          else:
            final_forecasts.append(f.median)

      if self.standardize:
        final_forecasts = self.data_manager.unstandardize_data(final_forecasts)
        test = self.data_manager.unstandardize_data(test)

      mase_metrics = []
      for item_id, ts in enumerate(test):
        training_data = ts["target"].T[:-self.prediction_length]
        ground_truth = ts["target"].T[-self.prediction_length:]

        y_pred_naive = np.array(training_data)[:-int(self.seasonality)]
        mae_naive = mean_absolute_error(np.array(training_data)[int(self.seasonality):], y_pred_naive, multioutput="uniform_average")

        if self.model == 'mqcnn' and self.batch_norm: 
           final_forecasts[item_id] = final_forecasts[item_id].reshape(-1)

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
      if not save:
        return np.mean(mase_metrics)
      
      # make directory called saved_nonorm_{self.model}_{self.dataset_name}
      dir_name = f'saved_nonorm_{self.model}_{self.dataset_name}_{self.standardize}'
      os.makedirs(dir_name, exist_ok=True)
      print('#####__DEBUG__######', np.mean(mase_metrics))
      return np.mean(mase_metrics), predictor, dir_name, history



def run(DATASET_NAME, model_choice, ctx, DATASET_FILE_FOLDER,  n_trials, multivariate, standardize, batch_norm, mean_scaling):

    start_time = time.perf_counter()
    study = optuna.create_study(direction="minimize")
    obj = Objective(
            model_choice,DATASET_NAME, ctx, DATASET_FILE_FOLDER, 
            multivariate=multivariate, standardize=standardize, batch_norm=batch_norm, mean_scaling=mean_scaling
        )
    study.optimize(
        obj,
        n_trials=n_trials,
    )

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    print(time.perf_counter() - start_time)

    if model_choice == 'feedforward':
      trial.params["num_hidden_dimensions"] = [ trial.params[f"hidden_dim_{i}"] for i in range(trial.params["num_layers"]) ]
    elif model_choice == 'transformer':
      trial.params["model_dim"] = trial.params["model_dim_num_heads_pair"][0]
      trial.params["num_heads"] = trial.params["model_dim_num_heads_pair"][1]
    results = []
    params_sets = []
    for i in range(5):
      res, predictor, dir_name, history = obj.train_and_test(trial.params, save=True)
      # plot and save training history

      plt.plot(history.loss_history, label='Training Loss')
      plt.plot(history.validation_loss_history, label='Validation Loss')
      plt.xlabel('Epoch')
      plt.ylabel('Loss')
      plt.title('Learning Curve')
      plt.legend()
      plt.grid(True)

      # Save the figure
      plt.savefig(f'{dir_name}/learning_curve_{i}.png')
      # save the history values
      with open(f'{dir_name}/loss_history_{i}.json', "w") as f:
        json.dump(history.loss_history, f)
      # Clear the current figure
      plt.clf()
      results.append(res)
      params_sets.append(trial.params)



    mean = np.array(results).mean()
    std = np.std(np.array(results))
    # print full results to file
    # with open(f'debug_results.txt', "w") as f:
    #     f.write(f'##### MEAN: {mean} STD: {std}\n')
    #     f.write(f'##### TRIAL: {trial.value} PARAMS: {trial.params}\n')
    #     # f.write(f'##### ALL RESULTS: {results}\n')
    #     # print results along with params for the trial
    #     for i in range(len(results)):
    #         f.write(f'##### RESULT: {results[i]} PARAMS: {params_sets[i]}\n')
    print(f'##### MEAN: {mean} STD: {std}')

    trial.params["mase_mean"] = mean
    trial.params["mase_std"] = std
    # save best params to json
    with open(f'{dir_name}/params.json', "w") as f:
        json.dump(trial.params, f)

    # save the last predictor
    os.makedirs(f'{dir_name}/predictor', exist_ok=True)
    predictor.serialize(Path(f"{dir_name}/predictor"))

    end_time = time.perf_counter()
    runtime = (end_time - start_time) / 60
    file_path = "output.txt"
    with open(file_path, "a") as file:
        if batch_norm:
          file.write(f' ########################### {model_choice} BATCH no norm {n_trials} trials on {DATASET_NAME} Final MASE: {trial.value}\n')
        elif mean_scaling:
          file.write(f' ########################### {model_choice} MEAN SCALING no norm {n_trials} trials on {DATASET_NAME} Final MASE: {trial.value}\n')
        elif standardize:
          file.write(f' ########################### {model_choice} STANDARDIZED no norm {n_trials} trials on {DATASET_NAME} Final MASE: {trial.value}\n')
        else:
           file.write(f' ########################### {model_choice} DEFAULT no norm {n_trials} trials on {DATASET_NAME} Final MASE: {trial.value}\n')
        
        file.write(f'with mean: {mean} and std: {std}\n')
        file.write(f'runtime: {runtime}\n')

    print("\n###FINISHED!###")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset_name', type=str)
  parser.add_argument('--root_folder', type=str)
  parser.add_argument('--model_choice', type=str)
#   parser.add_argument('--prediction_length', type=int)
#   parser.add_argument('--context_length', type=int)
  parser.add_argument('--ctx', type=str)
  parser.add_argument('--n_trials', type=int, default=20)
  parser.add_argument('--dataset_file_folder', type=str, default=None)
  parser.add_argument('--multivariate', action='store_true') # omit = false, --multivariate = true
  parser.add_argument('--standardize', action='store_true') # omit = false, --standardize = true
  parser.add_argument('--batch_norm', action='store_true') # omit = false, --batch_norm = true
  parser.add_argument('--mean_scaling', action='store_true') # omit = false, --mean_scaling = true
  # parser.add_argument('--use_tsf', action='store_true')
  args = parser.parse_args()
  print(args)
  run(args.dataset_name, 
      args.model_choice, 
      args.ctx, 
      args.dataset_file_folder, 
      args.n_trials, 
      args.multivariate, 
      args.standardize, 
      args.batch_norm,
      args.mean_scaling)