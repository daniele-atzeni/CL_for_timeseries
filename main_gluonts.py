from gluonts.dataset.repository import get_dataset

from gluonts.mx import Trainer  # SimpleFeedForwardEstimator


from my_simple_feedforward_joint_tr._estimator import SimpleFeedForwardEstimator
from normalizer import GASSimpleGaussian
from denormalizer import SumDenormalizer

import torch

dataset = get_dataset("australian_electricity_demand")

len_to_pred = (
    dataset.metadata.prediction_length if dataset.metadata.prediction_length else 50
)
CONTEXT_LENGTH = 2 * len_to_pred
n_features = len(list(dataset.train))

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

normalizer = GASSimpleGaussian()
normalizer.warm_up(train_ts, test_ts)

########
# let's add the means as new features of the time series
train_dataset = list(dataset.train)
test_dataset = list(dataset.test)
for i in range(normalizer.mus.shape[1]):
    new_feature = train_dataset[0]
    new_feature["target"] = normalizer.mus[:, i]
    train_dataset.append(new_feature)
########

input_dim = CONTEXT_LENGTH * n_features
output_dim = len_to_pred * n_features
denormalizer = SumDenormalizer(input_dim, output_dim)
# get weights and biases from the denormalizer
weight = denormalizer.mus_encoder.weight.detach().numpy()
bias = denormalizer.mus_encoder.bias.detach().numpy()
print(
    f"Retrieved weight and bias from the denormalizer with shape: {weight.shape}, {bias.shape}"
)
# last shape of hidden dimensions must be the same as the output dimension!
num_hidden_dimensions = [128, 64]
estimator = SimpleFeedForwardEstimator(
    weight,
    bias,
    n_features,
    num_hidden_dimensions=num_hidden_dimensions,
    prediction_length=dataset.metadata.prediction_length,
    context_length=CONTEXT_LENGTH,
    trainer=Trainer(ctx="cpu", epochs=5, learning_rate=1e-3, num_batches_per_epoch=100),
)

predictor = estimator.train(dataset.train)


from gluonts.evaluation import make_evaluation_predictions

forecast_it, ts_it = make_evaluation_predictions(
    dataset=dataset.test,  # test dataset
    predictor=predictor,  # predictor
    num_samples=100,  # number of sample paths we want for evaluation
)
