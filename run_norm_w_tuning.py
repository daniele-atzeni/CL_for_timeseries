import os
import pickle
import warnings
warnings.filterwarnings("ignore")
import numpy as np

from utils import get_dataset_from_file
from run_experiment.gas_experiment import run_gas_experiment
from default_parameters import *

from gluonts.mx.distribution import StudentTOutput, MultivariateGaussianOutput
from gluonts.evaluation import make_evaluation_predictions, Evaluator, MultivariateEvaluator
from gluonts.mx.trainer import Trainer
from gluonts.mx.trainer.callback import TrainingHistory
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt
import json
from pathlib import Path
from gluonts.dataset.split import split
from gluonts.dataset.common import ListDataset

import mxnet as mx
# from my_models.gluonts_models.univariate.feedforward_linear_means._estimator import (
#     SimpleFeedForwardEstimator as FF_gluonts_univariate_linear,
# )
from my_models.gluonts_models.univariate.probabilistic_forecast.feedforward_gas_means._estimator import (
    SimpleFeedForwardEstimator as FF_gluonts_univariate_gas,
)
from my_models.gluonts_models.feedforward_multivariate_linear_means._estimator import (
    SimpleFeedForwardEstimator as FF_gluonts_multivariate_linear,
)
from my_models.gluonts_models.univariate.point_forecast.feedforward_gas_means._estimator import (
    SimpleFeedForwardEstimator as FF_gluonts_univariate_gas_point,
)
# from my_models.gluonts_models.multivariate_feedforward_gas_means._estimator import (
#     SimpleFeedForwardEstimator as FF_gluonts_multivariate_gas,
# )

from my_models.gluonts_models.univariate.probabilistic_forecast.deepar_gas_means._estimator import (
    DeepAREstimator as DeepAR_gluonts_univariate_gas,
)
from my_models.gluonts_models.univariate.wavenet_gas_means._estimator import (
    WaveNetEstimator as WaveNet_gluonts_gas_means,
)
from my_models.gluonts_models.univariate.probabilistic_forecast.transformer_gas_means._estimator import (
    TransformerEstimator as Transformer_gluonts_gas_means,
)
from my_models.gluonts_models.univariate.point_forecast.transformer_gas_means._estimator import (
    TransformerEstimator as Transformer_gluonts_gas_means_point,
)
from my_models.gluonts_models.univariate.probabilistic_forecast.transformer_test._estimator import (
    TransformerEstimator as Transformer_test,
)
from my_models.gluonts_models.univariate.probabilistic_forecast.seq2seq._mq_dnn_estimator import (
    MQCNNEstimator as MQCNN_gluonts_univariate_gas,
)

from run_experiment.dl_model_experiment import GasHybridBlock
from normalizer import GASNormalizer
from sklearn.linear_model import LinearRegression

import time
import optuna
import argparse
import copy

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

    def __init__( self, MODEL, DATASET_NAME, ctx, DATASET_FILE_FOLDER, multivariate=False, mean_str=0.1, var_str=0.1):
        # self.train, self.test, self.freq, self.seasonality = get_dataset_from_file(f'{ROOT_FOLDER}/tsf_data/{DATASET_NAME}',prediction_length,context_length)
        self.model = MODEL
        self.multivariate = multivariate
        self.ctx = ctx
        self.dataset_name = DATASET_NAME

        

        DATASET_TYPE = "gluonts"  # "synthetic"
        DATASET_PARAMS = real_world_data_params  # synthetic_generation_params
        DATASET_PARAMS["multivariate"] = self.multivariate
        # DATASET_FILE_FOLDER = None #'tsf_data'

        NORMALIZER_NAME = "gas_t_student"  # "gas_simple_gaussian", "gas_complex_gaussian"
        NORMALIZER_INITIAL_GUESSES = gas_t_stud_initial_guesses  # gas_{name}_*
        NORMALIZER_BOUNDS = gas_t_stud_bounds
        NORMALIZER_PARAMS = gas_t_stud_params
        NORMALIZER_PARAMS['mean_strength'] = mean_str
        NORMALIZER_PARAMS['var_strength'] = var_str

        self.mean_str = mean_str
        self.var_str = var_str


        MEAN_LAYER_NAME = "gas"  # TODO: gas
        MEAN_LAYER_PARAMS = gas_mean_layer_params

        DL_MODEL_LIBRARY = "gluonts"  # "torch"
        DL_MODEL_NAME = MODEL  # TODO: "transformer"
        if self.model == 'feedforward':
            DL_MODEL_PARAMS = gluonts_feedforward_params
        elif self.model == 'transformer':
            DL_MODEL_PARAMS = gluonts_transformer_params
        elif self.model == 'deepar':
            DL_MODEL_PARAMS = gluonts_deepar_params
        elif self.model == 'wavenet':
            DL_MODEL_PARAMS = gluonts_wavenet_params
        elif self.model == 'mqcnn':
            DL_MODEL_PARAMS = gluonts_mqcnn_params

        N_TRAINING_SAMPLES = 5000
        N_TEST_SAMPLES = 1000

        ROOT_FOLDER = (
            f"RESULTS_{DATASET_NAME}_{self.model}_{NORMALIZER_NAME}_{MEAN_LAYER_NAME}_{DL_MODEL_LIBRARY}_{float(mean_str)}_{float(var_str)}"
        )
        if DATASET_PARAMS["multivariate"]:
            ROOT_FOLDER += "_multivariate"

        # run for the first time to get all the means and stuff
        training_params = run_gas_experiment(
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
            probabilistic=True,
        )

        # get the parameters needed to run the model part

        self.n_features, self.context_length, self.prediction_length, self.freq, self.dataset, self.mean_layer, self.dl_model_name, self.dl_model_params, self.folders = training_params
        self.seasonality = SEASONALITY_MAP[self.freq]
        if isinstance(self.seasonality, list):
          self.seasonality = min(self.seasonality) # Use to calculate MASE

        self.train_original, self.test = self.dataset
        # FOR TESTING
        # from data_manager import GluonTSDataManager
        # data_manager = GluonTSDataManager(DATASET_NAME, multivariate, DATASET_FILE_FOLDER, False)
        # self.train_original, self.test = data_manager.train_dataset, data_manager.test_dataset
        # new_train = []
        # for ts in self.train_original:
        #     new_train.append(
        #         {
        #             "start": ts["start"],
        #             "target": ts["target"],
        #             "feat_dynamic_real": np.zeros_like(ts["feat_dynamic_real"]),
        #             "feat_static_real": np.zeros_like(ts["feat_static_real"]),
        #         }
        #     )
        # self.train_original = new_train
        # new_test = []
        # for ts in self.test:
        #     new_test.append(
        #         {
        #             "start": ts["start"],
        #             "target": ts["target"],
        #             "feat_dynamic_real": np.zeros_like(ts["feat_dynamic_real"]),
        #             "feat_static_real": np.zeros_like(ts["feat_static_real"]),
        #         }
        #     )
        # self.test = new_train

        self.train, test_template = split(self.train_original, offset=-self.prediction_length)
        validation = test_template.generate_instances(
            prediction_length=self.prediction_length,
        )
        # Assuming `validation` is a list of (input, output) pairs
        # validation_data = [
        #     {
        #         "start": v[0]["start"],  # replace with the actual start time
        #         "target": np.concatenate([v[0]['target'], v[1]['target']]),
        #         "means_vars": v[0]["means_vars"],
        #         "gas_params": v[0]["gas_params"],
                
        #     }
        #     for v in validation
        # ]
        validation_data = []
        for v in validation:
            new_dict = {}
            for k in v[0].keys():
                if k != "target":
                    new_dict[k] = v[0][k]
                else:
                    new_dict[k] = np.concatenate([v[0][k], v[1][k]])
            validation_data.append(new_dict)

        self.validation = ListDataset(validation_data, freq=self.freq)

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
        elif self.model == 'deepar':
           return {
              "num_cells": trial.suggest_int("num_cells", 10, 100),
              "num_layers": trial.suggest_int("num_layers", 1, 10), # was 1, 5
              "trainer:learning_rate": trial.suggest_loguniform("trainer:learning_rate", 1e-6, 1e-3), # was 1e-6, 1e-4
              "trainer:epochs": trial.suggest_int("trainer:epochs", 10, 100)
           }
        elif self.model == 'mqcnn':
          return {
              "trainer:learning_rate": trial.suggest_loguniform("trainer:learning_rate", 1e-6, 1e-3),
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
              "trainer:learning_rate": trial.suggest_loguniform("trainer:learning_rate", 1e-6, 1e-3),
              "trainer:epochs": trial.suggest_int("trainer:epochs", 10, 100),
          }

    def __call__(self, trial):

        params = self.get_params(trial)

        # with open(f'train_{trial.number}_2.pkl', 'wb') as f:
        #     pickle.dump(self.train, f)


        return self.train_and_test(params)

    def train_and_test(self, params, save=False):
        

        history = TrainingHistory()
        trained_mean_layer = self.mean_layer
        if isinstance(trained_mean_layer, LinearRegression):
            mean_layer = mx.gluon.nn.HybridSequential()
            mean_layer.add(
                mx.gluon.nn.Dense(
                    units=self.prediction_length * self.n_features,
                    weight_initializer=mx.init.Constant(trained_mean_layer.coef_),
                    bias_initializer=mx.init.Constant(trained_mean_layer.intercept_),  # type: ignore # bias is a numpy array, don't know the reasons for this typing error
                )
            )
            mean_layer.add(
                mx.gluon.nn.HybridLambda(
                    lambda F, o: F.reshape(
                        o, (-1, self.prediction_length * self.n_features)
                    )  # no need for that but just to be sure
                )
            )
        elif isinstance(trained_mean_layer, GASNormalizer):
            mean_layer = GasHybridBlock(trained_mean_layer, self.n_features, self.prediction_length)
        else:
            raise ValueError(
                f"Unknown mean layer type: {type(trained_mean_layer)} {trained_mean_layer}"
            )

        # freeze the parameters
        for param in mean_layer.collect_params().values():
            param.grad_req = "null"

        # if self.model == 'feedforward' and self.multivariate:
        #     if isinstance(trained_mean_layer, GASNormalizer):
        #         estimator = FF_gluonts_multivariate_gas(
        #             mean_layer,
        #             self.n_features,
        #             MultivariateGaussianOutput(dim=self.n_features),
        #             prediction_length=self.prediction_length,
        #             context_length=self.context_length,
        #             num_hidden_dimensions= params['num_hidden_dimensions'], #num_hidden_dimensions,
        #             trainer=Trainer(hybridize=False,ctx=self.ctx,epochs=params['trainer:epochs'], learning_rate=params['trainer:learning_rate'],
        #                              num_batches_per_epoch=100, callbacks=[history]),
                    
        #         )
        #     else:
        #         estimator = FF_gluonts_multivariate_linear(
        #             mean_layer,
        #             self.n_features,
        #             MultivariateGaussianOutput(dim=self.n_features),
        #             prediction_length=self.prediction_length,
        #             context_length=self.context_length,
        #             num_hidden_dimensions= params['num_hidden_dimensions'], #num_hidden_dimensions,
        #             trainer=Trainer(hybridize=False,ctx=self.ctx,epochs=params['trainer:epochs'], learning_rate=params['trainer:learning_rate'],
        #                              num_batches_per_epoch=100, callbacks=[history]),
        #         )
        if self.model == 'feedforward' and not self.multivariate:
            if isinstance(trained_mean_layer, GASNormalizer):
                estimator = FF_gluonts_univariate_gas(
                    mean_layer,
                    distr_output=StudentTOutput(),
                    prediction_length=self.prediction_length,
                    context_length=self.context_length,
                    num_hidden_dimensions= params['num_hidden_dimensions'], #num_hidden_dimensions,
                    trainer=Trainer(hybridize=False,ctx=self.ctx,epochs=params['trainer:epochs'], learning_rate=params['trainer:learning_rate'],
                                     num_batches_per_epoch=100, callbacks=[history]),
                )
            # else:
            #     estimator = FF_gluonts_univariate_linear(
            #         mean_layer,
            #         distr_output=StudentTOutput(),
            #         prediction_length=self.prediction_length,
            #         context_length=self.context_length,
            #         trainer=Trainer(hybridize=False,ctx=self.ctx,epochs=params['trainer:epochs'], learning_rate=params['trainer:learning_rate'],
            #                          num_batches_per_epoch=100, callbacks=[history]),
            #     )
        elif self.model == 'transformer':
        #   from point_distributions import LaplaceFixedVarianceOutput
        #   from gluonts.mx.model.transformer import TransformerEstimator
        #   from my_models.gluonts_models.univariate.probabilistic_forecast.transformer_test2._estimator import (
        #     TransformerEstimator as Transformer_test2,
        #     )
        #   estimator = Transformer_gluonts_gas_means( # Transformer_gluonts_gas_means Transformer_gluonts_gas_means_point
        #       mean_layer,
        #       freq=self.freq,
        #       context_length=self.context_length,
        #       prediction_length=self.prediction_length,
        #       distr_output=StudentTOutput(), # StudentTOutput LaplaceFixedVarianceOutput
        #       inner_ff_dim_scale= params['inner_ff_dim_scale'],
        #       model_dim= params['model_dim'],
        #       embedding_dimension= params['embedding_dimension'],
        #       num_heads= params['num_heads'],
        #       dropout_rate= params['dropout_rate'],
        #       trainer=Trainer(hybridize=False,ctx=self.ctx,epochs=params['trainer:epochs'], learning_rate=params['trainer:learning_rate'],
        #                              num_batches_per_epoch=100, callbacks=[history]),
        #   )
        
          estimator = Transformer_gluonts_gas_means( # Transformer_gluonts_gas_means Transformer_gluonts_gas_means_point Transformer_test
              mean_layer,
              freq=self.freq,
              context_length=self.context_length,
              prediction_length=self.prediction_length,
              distr_output=StudentTOutput(), # StudentTOutput LaplaceFixedVarianceOutput
              scaling=False,
            #   inner_ff_dim_scale= params['inner_ff_dim_scale'],
            #   model_dim= params['model_dim'],
            #   embedding_dimension= params['embedding_dimension'],
            #   num_heads= params['num_heads'],
            #   dropout_rate= params['dropout_rate'],
              trainer=Trainer(hybridize=False,ctx=self.ctx,epochs=params['trainer:epochs'], learning_rate=params['trainer:learning_rate'],
                                     num_batches_per_epoch=100, callbacks=[history]),
            #   trainer=Trainer(hybridize=False,ctx=self.ctx,epochs=1, learning_rate=5*1e-4,
            #                          num_batches_per_epoch=100, callbacks=[history]),
          )

        elif self.model == 'wavenet' and not self.multivariate and isinstance(trained_mean_layer, GASNormalizer):
            estimator = WaveNet_gluonts_gas_means(
                mean_layer,
                freq=self.freq,
                prediction_length=self.prediction_length,
                trainer=Trainer(ctx=self.ctx,epochs=params['trainer:epochs'], learning_rate=params['trainer:learning_rate'],
                                    num_batches_per_epoch=100, callbacks=[history], add_default_callbacks=False),
            )
        elif self.model == 'mqcnn':
            estimator = MQCNN_gluonts_univariate_gas(
                mean_layer,
                freq=self.freq,
                prediction_length=self.prediction_length,
                context_length=self.context_length,
                distr_output=StudentTOutput(),
                quantiles=None,
                scaling=False,
                trainer=Trainer(ctx=self.ctx,epochs=params['trainer:epochs'], learning_rate=params['trainer:learning_rate'],
                                    num_batches_per_epoch=100, callbacks=[history], hybridize=False),
            )
        elif self.model == 'deepar' and not self.multivariate and isinstance(trained_mean_layer, GASNormalizer):
          estimator = DeepAR_gluonts_univariate_gas(
              mean_layer,
              freq=self.freq,
              distr_output=StudentTOutput(),
              context_length=self.context_length,
              prediction_length=self.prediction_length,
            #   num_cells= params['num_cells'],
            #   num_layers= params['num_layers'],
              trainer=Trainer(hybridize=False,ctx=self.ctx,epochs=params['trainer:epochs'], learning_rate=params['trainer:learning_rate'],
                                     num_batches_per_epoch=100, callbacks=[history]),
            #   trainer=Trainer(ctx=self.ctx,epochs=50, learning_rate=1e-4, num_batches_per_epoch=100),
          )

        ## TRAIN
        predictor = estimator.train(self.train, self.validation)
        ## EVALUATE
        if not save:
            test = copy.deepcopy(self.validation) # no need to deep copy but for sanity
        else:
            test = copy.deepcopy(self.test)
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test,  # test dataset
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

        mase_metrics = []
        for item_id, ts in enumerate(test):
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

        
        print("MINE")
        print(np.mean(mase_metrics))

        # print("GLUONTS")
        
        # tss = list(ts_it)
        # if self.multivariate:
        #     evaluator = MultivariateEvaluator(quantiles=[0.1, 0.5, 0.9])
        # else:
        #     evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])

        # agg_metrics, item_metrics = evaluator(tss, forecasts)  # type: ignore # we are sure that tss is a list of DataFrame in multivariate case
        # # print(json.dumps(agg_metrics, indent=4))
        # # print(agg_metrics)
        # print(item_metrics.mean())

        if not save:
            return np.mean(mase_metrics)
      
        # make directory called saved_nonorm_{self.model}_{self.dataset_name}
        dir_name = f'saved_GAS_{self.model}_{self.dataset_name}_mean_str_{float(self.mean_str)}_var_str_{float(self.var_str)}'
        os.makedirs(dir_name, exist_ok=True)

        return np.mean(mase_metrics), predictor, dir_name, history


def run(DATASET_NAME, model_choice, ctx, DATASET_FILE_FOLDER, n_trials, mean_str, var_str):

    multivariate = False
    start_time = time.perf_counter()

    if mean_str == -1 and var_str == -1:
        mean_strs = [0, 0.5, 0.1, 0.01, 0.001]
        var_strs = [0, 0.5, 0.1, 0.01, 0.001]
    elif mean_str == 0 and var_str == 0:
        mean_strs = [0]
        var_strs = [0]
    best_best_trial = None
    best_str = None
    trial_values = []
    for (mean_str, var_str) in zip(mean_strs, var_strs):
        study = optuna.create_study(direction="minimize")
        obj = Objective(
                model_choice,DATASET_NAME, ctx, DATASET_FILE_FOLDER, multivariate, mean_str, var_str
            )
        study.optimize(
            obj,
            n_trials=n_trials,
        )

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))
        trial_values.append({'str': (mean_str,var_str), 'best_valid_value':trial.value})

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        print(time.perf_counter() - start_time)

        if best_best_trial is None or trial.value < best_best_trial.value:
            best_best_trial = trial
            best_str = (mean_str, var_str)
    trial = best_best_trial
    if model_choice == 'feedforward':
      trial.params["num_hidden_dimensions"] = [ trial.params[f"hidden_dim_{i}"] for i in range(trial.params["num_layers"]) ]
    elif model_choice == 'transformer':
      trial.params["model_dim"] = trial.params["model_dim_num_heads_pair"][0]
      trial.params["num_heads"] = trial.params["model_dim_num_heads_pair"][1]

    results = []
    obj = Objective(
                model_choice,DATASET_NAME, ctx, DATASET_FILE_FOLDER, multivariate, best_str[0], best_str[1]
            )
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



    mean = np.array(results).mean()
    std = np.std(np.array(results))
    print(f'##### MEAN: {mean} STD: {std}')

    trial.params["mase_mean"] = mean
    trial.params["mase_std"] = std
    # save best params to json
    with open(f'{dir_name}/params.json', "w") as f:
        json.dump(trial.params, f)

    # save the last predictor - SERIALIZATION DOESNT WORK
    # os.makedirs(f'{dir_name}/predictor', exist_ok=True)
    # predictor.serialize(Path(f"{dir_name}/predictor"))

    end_time = time.perf_counter()
    runtime = (end_time - start_time) / 60
    file_path = "output.txt"
    with open(file_path, "a") as file:
        file.write(f' ########################### {model_choice} with gas norm gas means {n_trials} trials on {DATASET_NAME} Final MASE: {trial.value}\n')
        file.write(f'trial values: {trial_values}\n')
        file.write(f'with mean_str: {best_str[0]} and var_str: {best_str[1]}\n')
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
  parser.add_argument('--mean_str', type=float, default=0.01)
  parser.add_argument('--var_str', type=float, default=0.01)
  parser.add_argument('--dataset_file_folder', type=str, default=None)
  # parser.add_argument('--use_tsf', action='store_true')
  args = parser.parse_args()
  print(args)
  run(args.dataset_name, args.model_choice, args.ctx, args.dataset_file_folder, args.n_trials, args.mean_str, args.var_str)
