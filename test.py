
import os
import pickle

import numpy as np

from gluonts.evaluation import make_evaluation_predictions, MultivariateEvaluator, Evaluator
from gluonts.mx.trainer import Trainer
from gluonts.mx.distribution import StudentTOutput, MultivariateGaussianOutput
from sklearn.metrics import mean_absolute_error
from gluonts.dataset.multivariate_grouper import MultivariateGrouper

from gluonts.mx.model.simple_feedforward import SimpleFeedForwardEstimator
from my_models.gluonts_models.ffn_multivar import SimpleFeedForwardEstimator as FF_gluonts_multivariate

from data_manager import GluonTSDataManager

from gluonts.mx.model.deepar import DeepAREstimator 
from gluonts.mx.model.transformer import TransformerEstimator


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

DATASET_NAME, multivariate, DATASET_FILE_FOLDER = ('fred_md', False, None)
data_manager = GluonTSDataManager(DATASET_NAME, multivariate, DATASET_FILE_FOLDER)
n_features = data_manager.n_features
context_length = data_manager.context_length
prediction_length = data_manager.prediction_length
train = data_manager.train_dataset
test = data_manager.test_dataset
seasonality = seasonality[DATASET_NAME]

estimator = DeepAREstimator(
    freq=data_manager.freq,
    prediction_length=prediction_length,
    context_length=context_length,
    distr_output=StudentTOutput(),
    trainer=Trainer(ctx='gpu',epochs=50, learning_rate=1e-4, num_batches_per_epoch=100),
)
estimator = TransformerEstimator(
    freq=data_manager.freq,
    context_length=context_length,
    prediction_length=prediction_length,
    trainer=Trainer(ctx='gpu',epochs=50, learning_rate=1e-4, num_batches_per_epoch=100),
)

predictor = estimator.train(train)

forecast_it, ts_it = make_evaluation_predictions(
    dataset=test,  # test dataset
    predictor=predictor,  # predictor
    num_samples=100,  # number of sample paths we want for evaluation
)

forecasts = list(forecast_it)

final_forecasts_multi = []
for f in forecasts:
    final_forecasts_multi.append(f.median)

mase_metrics = []
for item_id, ts in enumerate(test):
    training_data = ts["target"].T[:-prediction_length]
    ground_truth = ts["target"].T[-prediction_length:]

    y_pred_naive = np.array(training_data)[:-int(seasonality)]
    mae_naive = mean_absolute_error(np.array(training_data)[int(seasonality):], y_pred_naive, multioutput="uniform_average")

    mae_score = mean_absolute_error(
        np.array(ground_truth),
        final_forecasts_multi[item_id],
        sample_weight=None,
        multioutput="uniform_average",
    )

    epsilon = np.finfo(np.float64).eps
    if mae_naive == 0:
        continue
    mase_score = mae_score / np.maximum(mae_naive, epsilon)


    mase_metrics.append(mase_score)

print(np.mean(mase_metrics))