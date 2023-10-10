from gluonts.dataset.repository import get_dataset

from gluonts.mx import Trainer  # SimpleFeedForwardEstimator


from my_simple_feedforward_joint_tr._estimator import SimpleFeedForwardEstimator
from normalizer import GASSimpleGaussian
from denormalizer import SumDenormalizer

import torch
import numpy as np

dataset = get_dataset("australian_electricity_demand")
if dataset.test is None:
    raise ValueError("dataset.test is None")

# normalize the dataset and add means as features
# create tensor datasets
train_torch_list = [torch.from_numpy(el["target"]) for el in dataset.train]
max_len = max([ts.shape[0] for ts in train_torch_list])
for i, el in enumerate(train_torch_list):
    if el.shape[0] < max_len:
        n_el_to_add = max_len - el.shape[0]
        el_to_add = torch.zeros(n_el_to_add)
        train_torch_list[i] = torch.cat([el, el_to_add])
train_ts = torch.stack(train_torch_list, dim=1)

test_torch_list = [torch.from_numpy(el["target"]) for el in dataset.test]
max_len = max([ts.shape[0] for ts in test_torch_list])
for i, el in enumerate(test_torch_list):
    if el.shape[0] < max_len:
        n_el_to_add = max_len - el.shape[0]
        el_to_add = torch.zeros(n_el_to_add)
        test_torch_list[i] = torch.cat([el, el_to_add])
test_ts = torch.stack(test_torch_list, dim=1)
# initialize normalizer and compute means
normalizer = GASSimpleGaussian()
normalizer.warm_up(train_ts, test_ts)
# means and vars are in normalizer.mus and normalizer.vars (shape (ts_len, n_feat)), and are of the same length of the test dataset features
if normalizer.mus is None or normalizer.vars is None:
    raise ValueError("normalizer.mus or normalizer.vars is None")
means = normalizer.mus.detach().numpy()
vars = normalizer.vars.detach().numpy()


def normalize_dataset_feature(
    i: int, feature: dict, means: np.ndarray, vars: np.ndarray
):
    ts = feature["target"]

    feat_len = ts.shape[0]
    means = means[:feat_len]
    vars = vars[:feat_len]
    ts = (ts - means[:, i]) / np.sqrt(vars[:, i])

    feature["target"] = ts
    return feature


dataset_train = [
    normalize_dataset_feature(i, el, means, vars) for i, el in enumerate(dataset.train)
]
dataset_test = [
    normalize_dataset_feature(i, el, means, vars) for i, el in enumerate(dataset.test)
]


# add means to datasets as features
def add_means_to_dataset(dataset, means):
    for i in range(means.shape[1]):
        new_feature = dataset[0]
        len_ts = new_feature["target"].shape[0]
        new_feature["target"] = means[:len_ts, i]
        dataset.append(new_feature)
    return dataset


dataset_train = add_means_to_dataset(dataset_train, means)
dataset_test = add_means_to_dataset(dataset_test, means)

# end of dataset preprocessing, now train
len_to_pred = (
    dataset.metadata.prediction_length if dataset.metadata.prediction_length else 50
)
CONTEXT_LENGTH = 2 * len_to_pred
n_features = len(list(dataset.train))

num_hidden_dimensions = [128, 64]
estimator = SimpleFeedForwardEstimator(
    n_features=n_features,
    num_hidden_dimensions=num_hidden_dimensions,
    prediction_length=dataset.metadata.prediction_length,
    context_length=CONTEXT_LENGTH,
    trainer=Trainer(ctx="cpu", epochs=5, learning_rate=1e-3, num_batches_per_epoch=100),
)

predictor = estimator.train(dataset_train)


from gluonts.evaluation import make_evaluation_predictions

forecast_it, ts_it = make_evaluation_predictions(
    dataset=dataset_test,  # test dataset
    predictor=predictor,  # predictor
    num_samples=100,  # number of sample paths we want for evaluation
)
