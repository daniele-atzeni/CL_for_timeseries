from gluonts.dataset.repository import get_dataset
from gluonts.mx import Trainer
from gluonts.evaluation import make_evaluation_predictions, Evaluator

import mxnet as mx

from my_simple_feedforward._estimator import SimpleFeedForwardEstimator
from normalizer import GASSimpleGaussian, GASComplexGaussian
from utils_old import initialize_gluonts_dataset, create_forecasting_tensors

import numpy as np
from sklearn import linear_model
import json


# GET THE DATASET
print("Getting the dataset...")
DATASET_NAME = "car_parts_without_missing"
dataset = get_dataset(DATASET_NAME)
print("Done.")

# WE SUPPOSE THAT THE DATASET HAS TIME SERIES OF EQUAL LENGTHS
assert len(set([el["target"].shape[0] for el in dataset.train])) == 1, (
    "Time series of different lengths in the train dataset. "
    "This is not supported by the normalizer."
)

# INITIALIZE SOME PARAMETERS
assert (
    dataset.metadata.prediction_length is not None
), "Prediction length cannot be None"
prediction_length = dataset.metadata.prediction_length
context_length = 2 * prediction_length
assert dataset.metadata.freq is not None, "Frequency length cannot be None"
freq = dataset.metadata.freq

# n_features = len(list(dataset.train)) NOOOOOOOO it'a always 1
n_features = 1

# We must produce a new dataset, with normalized time series and means and vars as new features
# we are sure that downloaded GluonTS datasets have no other features
train_dataset = [el["target"] for el in dataset.train]
train_starts = [el["start"] for el in dataset.train]
assert dataset.test is not None, "Test dataset cannot be None"
test_dataset = [el["target"] for el in dataset.test]
test_starts = [el["start"] for el in dataset.test]

# INITIALIZE THE NORMALIZER AND NORMALIZE THE DATASET
# normalizer is supposed to work on the complete GluonTS dataset or on a list of tensors
print("Initializing the normalizer...")
eta_mean, eta_var = 0.2, 0.2
# normalizer = GASSimpleGaussian(eta_mean, eta_var)
normalizer = GASComplexGaussian()
# we must warm up the normalizer with the complete dataset (even if in this case
# there is no need for computing static parameters)
print("Warming up the normalizer...")
normalizer.warm_up(train_dataset, test_dataset)
print("Done.")
# now we can normalize the dataset
print("Normalizing the dataset...")
norm_ts_train_list = normalizer.normalize(train_dataset)
norm_ts_test_list = normalizer.normalize(test_dataset)
print("Done.")
# and retrieve means to pass them as new features. Remember, these are as long as test ts
means, _ = normalizer.get_means_and_vars()
tr_means = [m[:-prediction_length] for m in means]

# INITIALIZE THE NEW DATASET
# we must add means as new features of the time series
assert isinstance(norm_ts_train_list, list) and isinstance(
    norm_ts_test_list, list
), "Something went wrong with normalization"
print("Initializing new GluonTS dataset...")
train_dataset = initialize_gluonts_dataset(
    norm_ts_train_list, tr_means, freq, train_starts
)
test_dataset = initialize_gluonts_dataset(norm_ts_test_list, means, freq, test_starts)
print("Done.")

# Before initializing the Estimator, we need to train the linear layer
# because we must pass linear layer parameters to the Estimator __init__

# SPLIT THE DATASET IN ORDER TO TRAIN THE MEAN LAYER
print("Creating dataset for the mean linear layer...")
N_TRAINING_SAMPLES_PER_TS = 100
N_TEST_SAMPLES_PER_TS = 100
train_mean_list = []
for i, means_i in enumerate(tr_means):
    _, xs, ys = create_forecasting_tensors(
        means_i, context_length, prediction_length, N_TRAINING_SAMPLES_PER_TS
    )
    train_mean_list.append((xs, ys))
test_mean_list = []
for i, means_i in enumerate(means):
    _, xs, ys = create_forecasting_tensors(
        means_i, context_length, prediction_length, N_TEST_SAMPLES_PER_TS
    )
    test_mean_list.append((xs, ys))
# initialize the datasets
train_means_xs = np.concatenate([el[0].numpy() for el in train_mean_list]).squeeze()
train_means_ys = np.concatenate([el[1].numpy() for el in train_mean_list]).squeeze()
test_means_xs = np.concatenate([el[0].numpy() for el in test_mean_list]).squeeze()
test_means_ys = np.concatenate([el[1].numpy() for el in test_mean_list]).squeeze()
print("Done.")

# INITIALIZE AND FIT THE REGRESSOR
print("Fitting the mean linear layer...")
regr = linear_model.LinearRegression()
regr.fit(train_means_xs, train_means_ys)
print(f"Score of the mean linear layer: {regr.score(test_means_xs, test_means_ys)}")
weight = regr.coef_
bias = regr.intercept_
assert isinstance(bias, np.ndarray), "bias must be a numpy array"

# INITIALIZE MXNET LINEAR MEAN LAYER
mean_layer = mx.gluon.nn.HybridSequential()
mean_layer.add(
    mx.gluon.nn.Dense(
        units=prediction_length,
        weight_initializer=mx.init.Constant(weight),
        bias_initializer=mx.init.Constant(bias),  # type: ignore # bias is a numpy array, don't know the reasons for this typing error
    )
)
mean_layer.add(
    mx.gluon.nn.HybridLambda(
        lambda F, o: F.reshape(
            o, (-1, prediction_length)
        )  # no need for that but just to be sure
    )
)
# freeze the parameters
for param in mean_layer.collect_params().values():
    param.grad_req = "null"

# INITIALIZE THE ESTIMATOR
print("Initializing the estimator...")
num_hidden_dimensions = [128, 17]
estimator = SimpleFeedForwardEstimator(
    mean_layer,
    num_hidden_dimensions=num_hidden_dimensions,
    prediction_length=prediction_length,
    context_length=context_length,
    trainer=Trainer(epochs=5, learning_rate=1e-3, num_batches_per_epoch=100),
)
print("Done.")

# TRAIN
predictor = estimator.train(train_dataset)

# EVALUATE
forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_dataset,  # test dataset
    predictor=predictor,  # predictor
    num_samples=100,  # number of sample paths we want for evaluation
)

forecasts = list(forecast_it)
tss = list(ts_it)

evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
agg_metrics, item_metrics = evaluator(tss, forecasts)

print(json.dumps(agg_metrics, indent=4))
print(item_metrics.head())


#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# OLD CODE FOR TENSOR LINEAR REGRESSOR

"""
import torch
from torch.utils.data import TensorDataset, DataLoader

# train
train_means_xs = torch.cat([el[0] for el in train_mean_list])
train_means_ys = torch.cat([el[1] for el in train_mean_list])
train_tensor_dataset = TensorDataset(train_means_xs, train_means_ys)
# test
test_means_xs = torch.cat([el[0] for el in test_mean_list])
test_means_ys = torch.cat([el[1] for el in test_mean_list])
test_tensor_dataset = TensorDataset(test_means_xs, test_means_ys)

# INITIALIZE MEAN LAYER OF THE DENORMALIZER
input_dim = context_length * n_features
output_dim = prediction_length * n_features
mean_layer = torch.nn.Linear(input_dim, output_dim)

# TRAIN DENORMALIZER MEAN LAYER
OPTIM = torch.optim.Adam(mean_layer.parameters(), lr=1e-3)
CRITERION = torch.nn.MSELoss()
BATCH_SIZE = 32
train_dataloader = DataLoader(train_tensor_dataset, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_tensor_dataset)
EPOCHS = 5
for epoch in range(EPOCHS):
    epoch_loss = 0
    for xs, ys in train_dataloader:
        # xs are (batch, context_length, n_features)
        # ys are (batch, prediction_length, n_features)
        OPTIM.zero_grad()
        # forward pass
        xs = xs.reshape((xs.shape[0], -1))
        pred = mean_layer(xs).reshape(ys.shape)
        # compute loss
        loss = CRITERION(pred, ys)
        # backward pass
        loss.backward()
        # update parameters
        OPTIM.step()

        epoch_loss += loss.item()
    print(f"Epoch {epoch} - loss: {epoch_loss}")
# compute losses on the tests
loss = 0
for i, (xs, ys) in enumerate(test_dataloader):
    xs = xs.reshape((xs.shape[0], -1))
    pred = mean_layer(xs).reshape(ys.shape)
    loss += CRITERION(pred, ys).item()
print(f"total loss: {loss}, avg loss: {loss / len(test_tensor_dataset)}")
# retrieve weights
weight = mean_layer.weight.detach().numpy()
bias = mean_layer.bias.detach().numpy()
"""
