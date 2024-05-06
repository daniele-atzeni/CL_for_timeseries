import os

import numpy as np

from gluonts.dataset import DataEntry
from gluonts.dataset.repository import get_dataset as gluonts_get_dataset
from gluonts.dataset.common import ListDataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from utils import get_dataset_from_file

from torch import from_numpy
from torch.utils.data import TensorDataset

from utils import get_dataset_from_file

TSDataset = list[np.ndarray]

NAME_TO_CONTEXT_AND_PRED = {
    "nn5_weekly_dataset": (65, 8),
    "us_births_dataset": (9, 30),
    "solar_10_minutes_dataset": (50, 1008),
    "weather_dataset": (9, 30),
    "sunspot_dataset_without_missing_values": (9, 30),
}


class GluonTSDataManager:
    def __init__(
        self, name: str, multivariate: bool, root_folder: str | None = None, prediction_length: int | None = None
    ) -> None:
        """
        Initialize the data manager. The stored train and test time series are
        lists of dict with the "target" field being either 1D array (univariate)
        or 2D array (n_feat, ts_length) (multivariate).
        The data obtained by the normalizer are lists of numpy arrays of shape
        (ts_length, n_features), with n_features = 1 for univariate.
        """
        # self.MIN_VALUE_TO_SCALE = 100

        self.name = name
        self.multivariate = multivariate
        self.init_main_dataset(root_folder, prediction_length=prediction_length)
        # data from normalizer
        self.train_means = None
        self.train_vars = None
        self.test_means = None
        self.test_vars = None
        self.train_params = None

    def init_main_dataset(self, root_folder: str | None, prediction_length: int | None = None) -> None:
        """
        This method must initialize:
        - self.train_dataset
        - self.test_dataset
        - self.n_features: the number of features of the dataset
        - self.prediction_length: the prediction length of the dataset
        - self.context_length: the context length of the dataset
        - self.freq: the frequency of the dataset
        Train and test datasets are GluonTS datasets with the target field being
        either 1D array (univariate) or 2D array (n_feat, ts_length) (multivariate).
        """
        if root_folder is None:
            regenerate = True if prediction_length is not None else False
            gluonts_dataset = gluonts_get_dataset(self.name, prediction_length=prediction_length, regenerate=regenerate)
            assert gluonts_dataset.test is not None, "No test dataset"
            # remove all the features we won't use
            # we need only the target, because we will use other fields for
            # means and vars
            #train_dataset = []
            #for el in gluonts_dataset.train:
            #    train_dataset.append({"target": el["target"], "start": el["start"]})
            #test_dataset = []
            #for el in gluonts_dataset.test:
            #    test_dataset.append({"target": el["target"], "start": el["start"]})
            train_dataset = list(gluonts_dataset.train)
            test_dataset = list(gluonts_dataset.test)

            # train_dataset = train_dataset[:2]  # for debugging
            # test_dataset = test_dataset[:2]  # for debugging

            assert isinstance(gluonts_dataset.metadata.prediction_length, int)
            self.prediction_length = gluonts_dataset.metadata.prediction_length
            self.context_length = 2 * self.prediction_length
            self.freq = gluonts_dataset.metadata.freq
            self.seasonality = None
        else:
            context_length, external_forecast_horizon = NAME_TO_CONTEXT_AND_PRED[
                self.name
            ]
            train_dataset, test_dataset, freq, seasonality = get_dataset_from_file(
                os.path.join(root_folder, self.name),
                external_forecast_horizon,
                context_length,
            )
            self.prediction_length = external_forecast_horizon
            self.context_length = context_length
            self.freq = freq
            self.seasonality = seasonality

        train_dataset = self._scale_dataset(train_dataset)
        test_dataset = self._scale_dataset(test_dataset)

        self.n_features = len(list(train_dataset)) if self.multivariate else 1
        assert test_dataset is not None
        if self.multivariate:
            assert (
                len(list(train_dataset)) > 1
            ), "Just one time series in the dataset, it's impossible to make it multivariate"
            train_grouper = MultivariateGrouper(max_target_dim=self.n_features)
            test_grouper = MultivariateGrouper(
                max_target_dim=self.n_features,
                num_test_dates=len(list(test_dataset)) // self.n_features,
            )
            self.train_dataset = train_grouper(train_dataset)
            self.test_dataset = test_grouper(test_dataset)
        else:
            self.train_dataset = train_dataset
            self.test_dataset = test_dataset

    def _scale_dataset(self, dataset: list[DataEntry]) -> list[DataEntry]:
        """
        This method shrinks the dataset by dividing all the time series by the
        mean of the first self.context_length points. Here, all time series are
        univariate since they come from file or from GluonTS.
        """
        # if max([el["target"].max() for el in dataset]) < self.MIN_VALUE_TO_SCALE:
        #    return dataset

        new_dataset = []
        for el in dataset:
            scale_val = np.mean(el["target"][: self.context_length])
            el["target"] = el["target"] / scale_val
            new_dataset.append(el)
        return new_dataset

    def get_dataset_for_normalizer(self) -> tuple[TSDataset, TSDataset]:
        """
        This method returns the dataset that will be used to train the normalizer.
        Normalizer expects a list of numpy arrays of shape (ts_length, n_features).
        This method must return both train and test datasets.
        """
        if self.multivariate:
            # multivariate time series are of shape (n_features, ts_length)
            # normalizer wants the opposite
            train_dataset = [el["target"].T for el in self.train_dataset]
            test_dataset = [el["target"].T for el in self.test_dataset]
        else:
            # univariate time series are 1D, normalizer wants 2D
            train_dataset = [
                np.expand_dims(el["target"], -1) for el in self.train_dataset
            ]
            test_dataset = [
                np.expand_dims(el["target"], -1) for el in self.test_dataset
            ]

        return train_dataset, test_dataset

    def set_data_from_normalizer(
        self,
        train_means: TSDataset,
        train_vars: TSDataset,
        test_means: TSDataset,
        test_vars: TSDataset,
        train_params: TSDataset,
    ) -> None:
        assert (
            len(train_means) == len(train_vars)
            and len(test_means) == len(test_vars)
            and len(train_means) == len(self.train_dataset)
            and len(test_means) == len(self.test_dataset)
        ), "Wrong data for normalizer"
        self.train_means = train_means
        self.train_vars = train_vars
        self.test_means = test_means
        self.test_vars = test_vars
        self.train_params = train_params

    def _prepare_for_sampling_ts(self, n_samples: int, phase: str, seed: int) -> tuple:
        assert phase in ["train", "test"], "Wrong phase"
        if phase == "train":
            dataset = [el["target"] for el in self.train_dataset]
            means = self.train_means
            vars = self.train_vars
        else:
            dataset = [el["target"] for el in self.test_dataset]
            means = self.test_means
            vars = self.test_vars
        assert means is not None, "Data from normalizer not set"

        np.random.seed(seed)
        n_samples_per_ts = n_samples // len(dataset)

        # computing starting indices
        start_indices = []
        for ts in dataset:
            # ts is shape (n_features, ts_length) or (ts_length)
            ts_length = ts.shape[-1]
            start_indices.append(
                np.random.randint(
                    low=0,
                    high=ts_length - self.context_length - self.prediction_length,
                    size=n_samples_per_ts,
                )
            )
        return dataset, means, vars, start_indices, n_samples_per_ts

    def _split_data_for_mean_layer(
        self, n_samples: int, phase: str, seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        (
            dataset,
            means,
            _,  # no need for vars in this case
            start_indices,
            n_samples_per_ts,
        ) = self._prepare_for_sampling_ts(n_samples, phase, seed)
        # init results
        mean_layer_x = np.empty((n_samples, self.context_length * self.n_features))
        mean_layer_y = np.empty((n_samples, self.prediction_length * self.n_features))

        # slice and fill the arrays
        for i, (ts, mean_ts, start_idxs) in enumerate(
            zip(dataset, means, start_indices)
        ):
            for j, start_idx in enumerate(start_idxs):
                # ts is shape (n_features, ts_length) or (ts_length)
                mean_window_x = mean_ts[start_idx : start_idx + self.context_length]
                mean_layer_x[i * n_samples_per_ts + j] = mean_window_x.reshape(
                    self.context_length * self.n_features
                )
                mean_window_y = ts[
                    ...,
                    start_idx
                    + self.context_length : start_idx
                    + self.context_length
                    + self.prediction_length,
                ]
                mean_layer_y[i * n_samples_per_ts + j] = mean_window_y.reshape(
                    self.prediction_length * self.n_features
                )

        return mean_layer_x, mean_layer_y

    def get_dataset_for_linear_mean_layer(
        self, n_training_samples: int, n_test_samples: int, seed: int = 42
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        This method returns the dataset that will be used to train the mean layer.
        Mean layer expects a list of numpy arrays of shape (ts_length, n_features)
        even in the univariate case, with n_features = 1.
        """
        train_x, train_y = self._split_data_for_mean_layer(
            n_training_samples, "train", seed
        )
        test_x, test_y = self._split_data_for_mean_layer(n_test_samples, "test", seed)
        return train_x, train_y, test_x, test_y

    def get_gluon_dataset_for_dl_layer(self) -> tuple[list[DataEntry], list[DataEntry]]:
        assert (
            self.train_means is not None
            and self.train_vars is not None
            and self.test_means is not None
            and self.test_vars is not None
            and self.train_params is not None
        ), "Data from normalizer not set"
        # self.params is a list of #GAS_params lists of numpy arrays of shape (n_features)

        if not self.multivariate:
            train_dataset = ListDataset(
                [
                    {
                        "target": data_entry["target"],
                        "start": data_entry["start"],
                        "means_vars": np.concatenate((mean, var), axis=1).T,
                        "gas_params": (
                            np.concatenate(train_param)
                            if isinstance(train_param, list)
                            else train_param
                        ),
                    }
                    for data_entry, mean, var, train_param in zip(
                        self.train_dataset,
                        self.train_means,
                        self.train_vars,
                        self.train_params,
                    )
                ],
                freq=self.freq,
            )
            test_dataset = ListDataset(
                [
                    {
                        "target": data_entry["target"],
                        "start": data_entry["start"],
                        "means_vars": np.concatenate((mean, var), axis=1).T,
                        "gas_params": (
                            np.concatenate(train_param)
                            if isinstance(train_param, list)
                            else train_param
                        ),
                    }
                    for data_entry, mean, var, train_param in zip(
                        self.test_dataset,
                        self.test_means,
                        self.test_vars,
                        self.train_params,
                    )
                ],
                freq=self.freq,
            )
        else:
            train_dataset = ListDataset(
                [
                    {
                        "target": data_entry["target"],
                        "start": data_entry["start"],
                        "means_vars": np.concatenate((mean, var), axis=1).T,
                        "gas_params": (
                            np.concatenate(train_param)
                            if isinstance(train_param, list)
                            else train_param
                        ),
                    }
                    for data_entry, mean, var, train_param in zip(
                        self.train_dataset,
                        self.train_means,
                        self.train_vars,
                        self.train_params,
                    )
                ],
                freq=self.freq,
                one_dim_target=False,
            )
            test_dataset = ListDataset(
                [
                    {
                        "target": data_entry["target"],
                        "start": data_entry["start"],
                        "means_vars": np.concatenate((mean, var), axis=1).T,
                        "gas_params": (
                            np.concatenate(train_param)
                            if isinstance(train_param, list)
                            else train_param
                        ),
                    }
                    for data_entry, mean, var, train_param in zip(
                        self.test_dataset,
                        self.test_means,
                        self.test_vars,
                        self.train_params,
                    )
                ],
                freq=self.freq,
                one_dim_target=False,
            )

        return train_dataset, test_dataset

    def _split_data_for_dl_layer(
        self, n_samples: int, phase: str, seed: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        (dataset, means, vars, start_indices, _) = self._prepare_for_sampling_ts(
            n_samples, phase, seed
        )

        # init results
        shape_x = (n_samples, self.context_length, self.n_features)
        shape_y = (n_samples, self.prediction_length, self.n_features)
        dl_layer_x = np.empty(shape_x)
        dl_means_x = np.empty(shape_x)
        dl_vars_x = np.empty(shape_x)
        dl_layer_y = np.empty(shape_y)

        # slice and fill the arrays
        for i, (ts, mean_ts, var_ts, start_idxs) in enumerate(
            zip(dataset, means, vars, start_indices)
        ):
            for start_idx in start_idxs:
                # ts is shape (n_features, ts_length) or (ts_length)
                # mean_ts is shape (ts_length, n_features)
                # var_ts is shape (ts_length, n_features)
                ts_window_x = ts[..., start_idx : start_idx + self.context_length]
                mean_window_x = mean_ts[start_idx : start_idx + self.context_length]
                var_window_x = var_ts[start_idx : start_idx + self.context_length]

                ts_window_y = ts[
                    ...,
                    start_idx
                    + self.context_length : start_idx
                    + self.context_length
                    + self.prediction_length,
                ]

                if self.multivariate:
                    ts_window_x = ts_window_x.T
                    ts_window_y = ts_window_y.T
                else:
                    ts_window_x = np.expand_dims(ts_window_x, -1)
                    ts_window_y = np.expand_dims(ts_window_y, -1)

                dl_layer_x[i] = ts_window_x
                dl_means_x[i] = mean_window_x
                dl_vars_x[i] = var_window_x
                dl_layer_y[i] = ts_window_y

        return dl_layer_x, dl_means_x, dl_vars_x, dl_layer_y

    def get_torch_dataset_for_dl_layer(
        self, n_training_samples: int, n_test_samples: int, seed: int = 42
    ) -> tuple[TensorDataset, TensorDataset]:
        """
        This method creates the dataset for the Torch DL model. Data for the DL
        model are of shape (n_samples, context_length, n_features) for x and
        (n_samples, prediction_length, n_features) for y.
        The model needs as x windows of:
        - the dataset
        - the means
        - the vars
        The model needs windows of the dataset as y.
        Dataset normalization takes place inside of the forward of the models.
        """
        train_x, train_mean, train_var, train_y = self._split_data_for_dl_layer(
            n_training_samples, "train", seed
        )
        test_x, test_mean, test_var, test_y = self._split_data_for_dl_layer(
            n_test_samples, "test", seed
        )

        train_dataset = TensorDataset(
            from_numpy(train_x),
            from_numpy(train_mean),
            from_numpy(train_var),
            from_numpy(train_y),
        )
        test_dataset = TensorDataset(
            from_numpy(test_x),
            from_numpy(test_mean),
            from_numpy(test_var),
            from_numpy(test_y),
        )
        return train_dataset, test_dataset


class SyntheticDatasetGetter(GluonTSDataManager):
    def __init__(self, name: str, multivariate: bool, generation_params) -> None:
        super().__init__(name, multivariate)
        self.generation_params = generation_params

    def get_main_dataset(self):
        pass

import os

import numpy as np

from gluonts.dataset import DataEntry
from gluonts.dataset.repository import get_dataset as gluonts_get_dataset
from gluonts.dataset.common import ListDataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from utils import get_dataset_from_file

# from torch import from_numpy
# from torch.utils.data import TensorDataset

from utils import get_dataset_from_file

TSDataset = list[np.ndarray]

NAME_TO_CONTEXT_AND_PRED = {
    "nn5_weekly_dataset": (65, 8),
    "us_births_dataset": (9, 30),
    "solar_10_minutes_dataset": (50, 1008),
    "weather_dataset": (9, 30),
    "sunspot_dataset_without_missing_values": (9, 30),
}
NAME_TO_CONTEXT_AND_PRED_GLUONTS = { # (context_length, prediction_length)
    "nn5_weekly": (65, 8),
    "solar_10_minutes": (50, 1008),
    "weather": (9, 30),
    "sunspot_without_missing": (9, 30),
    "hospital": (15, 12),
    'rideshare_without_missing': (210, 168),
    'fred_md': (15, 12)
}


class GluonTSDataManager:
    def __init__(
        self, name: str, multivariate: bool, root_folder: str | None = None, standardize: bool = False,
    ) -> None:
        """
        Initialize the data manager. The stored train and test time series are
        lists of dict with the "target" field being either 1D array (univariate)
        or 2D array (n_feat, ts_length) (multivariate).
        The data obtained by the normalizer are lists of numpy arrays of shape
        (ts_length, n_features), with n_features = 1 for univariate.
        """
        # self.MIN_VALUE_TO_SCALE = 100

        self.name = name
        self.multivariate = multivariate
        self.standardize = standardize
        self.scale_train_means = None
        self.scale_train_stds = None
        self.init_main_dataset(root_folder)
        # data from normalizer
        self.train_means = None
        self.train_vars = None
        self.test_means = None
        self.test_vars = None
        self.train_params = None


    def init_main_dataset(self, root_folder: str | None) -> None:
        """
        This method must initialize:
        - self.train_dataset
        - self.test_dataset
        - self.n_features: the number of features of the dataset
        - self.prediction_length: the prediction length of the dataset
        - self.context_length: the context length of the dataset
        - self.freq: the frequency of the dataset
        Train and test datasets are GluonTS datasets with the target field being
        either 1D array (univariate) or 2D array (n_feat, ts_length) (multivariate).
        """
        if root_folder is None:
            gluonts_dataset = gluonts_get_dataset(self.name)
            assert gluonts_dataset.test is not None, "No test dataset"
            # remove all the features we won't use
            # we need only the target, because we will use other fields for
            # means and vars
            #train_dataset = []
            #for el in gluonts_dataset.train:
            #    train_dataset.append({"target": el["target"], "start": el["start"]})
            #test_dataset = []
            #for el in gluonts_dataset.test:
            #    test_dataset.append({"target": el["target"], "start": el["start"]})
            train_dataset = list(gluonts_dataset.train)
            test_dataset = list(gluonts_dataset.test)

            # train_dataset = train_dataset[:1]  # for debugging
            # test_dataset = test_dataset[:1]  # for debugging

            assert isinstance(gluonts_dataset.metadata.prediction_length, int)
            if self.name in NAME_TO_CONTEXT_AND_PRED_GLUONTS.keys():
                self.context_length, self.prediction_length = NAME_TO_CONTEXT_AND_PRED_GLUONTS[self.name]
            else:
                self.prediction_length = gluonts_dataset.metadata.prediction_length
                self.context_length = 2 * self.prediction_length
            self.freq = gluonts_dataset.metadata.freq
            self.seasonality = None
        else:
            context_length, external_forecast_horizon = NAME_TO_CONTEXT_AND_PRED[
                self.name
            ]
            train_dataset, test_dataset, freq, seasonality = get_dataset_from_file(
                f'{root_folder}/{self.name}', external_forecast_horizon, context_length
            )
            self.prediction_length = external_forecast_horizon
            self.context_length = context_length
            self.freq = freq
            self.seasonality = seasonality

        # train_dataset = self._difference(train_dataset)
        # test_dataset = self._difference(test_dataset)

        train_dataset = self._scale_dataset(train_dataset)
        test_dataset = self._scale_dataset(test_dataset)
        
        if self.standardize:
            train_dataset, test_dataset = self._standardize_datasets(train_dataset, test_dataset)
            


        self.n_features = len(list(train_dataset)) if self.multivariate else 1
        assert test_dataset is not None
        if self.multivariate:
            assert (
                len(list(train_dataset)) > 1
            ), "Just one time series in the dataset, it's impossible to make it multivariate"
            train_grouper = MultivariateGrouper(max_target_dim=self.n_features)
            test_grouper = MultivariateGrouper(
                max_target_dim=self.n_features,
                num_test_dates=len(list(test_dataset)) // self.n_features,
            )
            self.train_dataset = train_grouper(train_dataset)
            self.test_dataset = test_grouper(test_dataset)
        else:
            self.train_dataset = train_dataset
            self.test_dataset = test_dataset

    def _standardize_datasets(self, train: list[DataEntry], test: list[DataEntry]) -> tuple[list[DataEntry], list[DataEntry]]:
        new_train = []
        new_test = []
        train_means = []
        train_stds = []
        for el in train:
            mean = np.mean(el["target"])
            std = np.std(el["target"])
            el["target"] = (el["target"] - mean) / std
            train_means.append(mean)
            train_stds.append(std)
            new_train.append(el)
        self.scale_train_means = train_means
        self.scale_train_stds = train_stds
        for el, mean, std in zip(test, train_means, train_stds):
            el["target"] = (el["target"] - mean) / std
            new_test.append(el)

        return new_train, new_test
    
    def unstandardize_data(self, data: list[any]) -> list[DataEntry]:
        new_data = []
        if type(data[0]) == dict:
            for el, mean, std in zip(data, self.scale_train_means, self.scale_train_stds):
                el["target"] = (el["target"] * std) + mean
                new_data.append(el)
        else:
            for el, mean, std in zip(data, self.scale_train_means, self.scale_train_stds):
                el = (el * std) + mean
                new_data.append(el)
        return new_data
    
    def _difference(self, dataset: list[DataEntry]) -> list[DataEntry]:
        differenced_dataset = []
        for data in dataset:
            target = data['target']
            differenced_target = target[1:] - target[:-1]
            differenced_data = {**data, 'target': differenced_target}
            differenced_dataset.append(differenced_data)
        return differenced_dataset
    
    def _scale_dataset(self, dataset: list[DataEntry]) -> list[DataEntry]:
        """
        This method shrinks the dataset by dividing all the time series by the
        mean of the first self.context_length points. Here, all time series are
        univariate since they come from file or from GluonTS.
        """
        # if max([el["target"].max() for el in dataset]) < self.MIN_VALUE_TO_SCALE:
        #    return dataset

        new_dataset = []
        for el in dataset:
            scale_val = np.mean(el["target"][: self.context_length])
            if scale_val == 0:
                scale_val = np.finfo(float).eps
            el["target"] = el["target"] / scale_val
            new_dataset.append(el)
        return new_dataset

    def get_dataset_for_normalizer(self) -> tuple[TSDataset, TSDataset]:
        """
        This method returns the dataset that will be used to train the normalizer.
        Normalizer expects a list of numpy arrays of shape (ts_length, n_features).
        This method must return both train and test datasets.
        """
        if self.multivariate:
            # multivariate time series are of shape (n_features, ts_length)
            # normalizer wants the opposite
            train_dataset = [el["target"].T for el in self.train_dataset]
            test_dataset = [el["target"].T for el in self.test_dataset]
        else:
            # univariate time series are 1D, normalizer wants 2D
            train_dataset = [
                np.expand_dims(el["target"], -1) for el in self.train_dataset
            ]
            test_dataset = [
                np.expand_dims(el["target"], -1) for el in self.test_dataset
            ]

        return train_dataset, test_dataset

    def set_data_from_normalizer(
        self,
        train_means: TSDataset,
        train_vars: TSDataset,
        test_means: TSDataset,
        test_vars: TSDataset,
        train_params: TSDataset,
    ) -> None:
        assert (
            len(train_means) == len(train_vars)
            and len(test_means) == len(test_vars)
            and len(train_means) == len(self.train_dataset)
            and len(test_means) == len(self.test_dataset)
        ), "Wrong data for normalizer"
        self.train_means = train_means
        self.train_vars = train_vars
        self.test_means = test_means
        self.test_vars = test_vars
        self.train_params = train_params

    def _prepare_for_sampling_ts(self, n_samples: int, phase: str, seed: int) -> tuple:
        assert phase in ["train", "test"], "Wrong phase"
        if phase == "train":
            dataset = [el["target"] for el in self.train_dataset]
            means = self.train_means
            vars = self.train_vars
        else:
            dataset = [el["target"] for el in self.test_dataset]
            means = self.test_means
            vars = self.test_vars
        assert means is not None, "Data from normalizer not set"

        np.random.seed(seed)
        n_samples_per_ts = n_samples // len(dataset)

        # computing starting indices
        start_indices = []
        for ts in dataset:
            # ts is shape (n_features, ts_length) or (ts_length)
            ts_length = ts.shape[-1]
            start_indices.append(
                np.random.randint(
                    low=0,
                    high=ts_length - self.context_length - self.prediction_length,
                    size=n_samples_per_ts,
                )
            )
        return dataset, means, vars, start_indices, n_samples_per_ts

    def _split_data_for_mean_layer(
        self, n_samples: int, phase: str, seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        (
            dataset,
            means,
            _,  # no need for vars in this case
            start_indices,
            n_samples_per_ts,
        ) = self._prepare_for_sampling_ts(n_samples, phase, seed)
        # init results
        mean_layer_x = np.empty((n_samples, self.context_length * self.n_features))
        mean_layer_y = np.empty((n_samples, self.prediction_length * self.n_features))

        # slice and fill the arrays
        for i, (ts, mean_ts, start_idxs) in enumerate(
            zip(dataset, means, start_indices)
        ):
            for j, start_idx in enumerate(start_idxs):
                # ts is shape (n_features, ts_length) or (ts_length)
                mean_window_x = mean_ts[start_idx : start_idx + self.context_length]
                mean_layer_x[i * n_samples_per_ts + j] = mean_window_x.reshape(
                    self.context_length * self.n_features
                )
                mean_window_y = ts[
                    ...,
                    start_idx
                    + self.context_length : start_idx
                    + self.context_length
                    + self.prediction_length,
                ]
                mean_layer_y[i * n_samples_per_ts + j] = mean_window_y.reshape(
                    self.prediction_length * self.n_features
                )

        return mean_layer_x, mean_layer_y

    def get_dataset_for_linear_mean_layer(
        self, n_training_samples: int, n_test_samples: int, seed: int = 42
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        This method returns the dataset that will be used to train the mean layer.
        Mean layer expects a list of numpy arrays of shape (ts_length, n_features)
        even in the univariate case, with n_features = 1.
        """
        train_x, train_y = self._split_data_for_mean_layer(
            n_training_samples, "train", seed
        )
        test_x, test_y = self._split_data_for_mean_layer(n_test_samples, "test", seed)
        return train_x, train_y, test_x, test_y

    def get_gluon_dataset_for_dl_layer(self) -> tuple[list[DataEntry], list[DataEntry]]:
        assert (
            self.train_means is not None
            and self.train_vars is not None
            and self.test_means is not None
            and self.test_vars is not None
            and self.train_params is not None
        ), "Data from normalizer not set"
        # self.params is a list of #GAS_params lists of numpy arrays of shape (n_features)

        if not self.multivariate:
            train_dataset = ListDataset(
                [
                    {
                        "target": data_entry["target"],
                        "start": data_entry["start"],
                        "means_vars": np.concatenate((mean, var), axis=1).T,
                        "gas_params": (
                            np.concatenate(train_param)
                            if isinstance(train_param, list)
                            else train_param
                        ),
                    }
                    for data_entry, mean, var, train_param in zip(
                        self.train_dataset,
                        self.train_means,
                        self.train_vars,
                        self.train_params,
                    )
                ],
                freq=self.freq,
            )
            test_dataset = ListDataset(
                [
                    {
                        "target": data_entry["target"],
                        "start": data_entry["start"],
                        "means_vars": np.concatenate((mean, var), axis=1).T,
                        "gas_params": (
                            np.concatenate(train_param)
                            if isinstance(train_param, list)
                            else train_param
                        ),
                    }
                    for data_entry, mean, var, train_param in zip(
                        self.test_dataset,
                        self.test_means,
                        self.test_vars,
                        self.train_params,
                    )
                ],
                freq=self.freq,
            )
        else:
            train_dataset = ListDataset(
                [
                    {
                        "target": data_entry["target"],
                        "start": data_entry["start"],
                        "means_vars": np.concatenate((mean, var), axis=1).T,
                        "gas_params": (
                            np.concatenate(train_param)
                            if isinstance(train_param, list)
                            else train_param
                        ),
                    }
                    for data_entry, mean, var, train_param in zip(
                        self.train_dataset,
                        self.train_means,
                        self.train_vars,
                        self.train_params,
                    )
                ],
                freq=self.freq,
                one_dim_target=False,
            )
            test_dataset = ListDataset(
                [
                    {
                        "target": data_entry["target"],
                        "start": data_entry["start"],
                        "means_vars": np.concatenate((mean, var), axis=1).T,
                        "gas_params": (
                            np.concatenate(train_param)
                            if isinstance(train_param, list)
                            else train_param
                        ),
                    }
                    for data_entry, mean, var, train_param in zip(
                        self.test_dataset,
                        self.test_means,
                        self.test_vars,
                        self.train_params,
                    )
                ],
                freq=self.freq,
                one_dim_target=False,
            )

        return train_dataset, test_dataset

    def _split_data_for_dl_layer(
        self, n_samples: int, phase: str, seed: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        (dataset, means, vars, start_indices, _) = self._prepare_for_sampling_ts(
            n_samples, phase, seed
        )

        # init results
        shape_x = (n_samples, self.context_length, self.n_features)
        shape_y = (n_samples, self.prediction_length, self.n_features)
        dl_layer_x = np.empty(shape_x)
        dl_means_x = np.empty(shape_x)
        dl_vars_x = np.empty(shape_x)
        dl_layer_y = np.empty(shape_y)

        # slice and fill the arrays
        for i, (ts, mean_ts, var_ts, start_idxs) in enumerate(
            zip(dataset, means, vars, start_indices)
        ):
            for start_idx in start_idxs:
                # ts is shape (n_features, ts_length) or (ts_length)
                # mean_ts is shape (ts_length, n_features)
                # var_ts is shape (ts_length, n_features)
                ts_window_x = ts[..., start_idx : start_idx + self.context_length]
                mean_window_x = mean_ts[start_idx : start_idx + self.context_length]
                var_window_x = var_ts[start_idx : start_idx + self.context_length]

                ts_window_y = ts[
                    ...,
                    start_idx
                    + self.context_length : start_idx
                    + self.context_length
                    + self.prediction_length,
                ]

                if self.multivariate:
                    ts_window_x = ts_window_x.T
                    ts_window_y = ts_window_y.T
                else:
                    ts_window_x = np.expand_dims(ts_window_x, -1)
                    ts_window_y = np.expand_dims(ts_window_y, -1)

                dl_layer_x[i] = ts_window_x
                dl_means_x[i] = mean_window_x
                dl_vars_x[i] = var_window_x
                dl_layer_y[i] = ts_window_y

        return dl_layer_x, dl_means_x, dl_vars_x, dl_layer_y

    # def get_torch_dataset_for_dl_layer(
    #     self, n_training_samples: int, n_test_samples: int, seed: int = 42
    # ) -> tuple[TensorDataset, TensorDataset]:
    #     """
    #     This method creates the dataset for the Torch DL model. Data for the DL
    #     model are of shape (n_samples, context_length, n_features) for x and
    #     (n_samples, prediction_length, n_features) for y.
    #     The model needs as x windows of:
    #     - the dataset
    #     - the means
    #     - the vars
    #     The model needs windows of the dataset as y.
    #     Dataset normalization takes place inside of the forward of the models.
    #     """
    #     train_x, train_mean, train_var, train_y = self._split_data_for_dl_layer(
    #         n_training_samples, "train", seed
    #     )
    #     test_x, test_mean, test_var, test_y = self._split_data_for_dl_layer(
    #         n_test_samples, "test", seed
    #     )

    #     train_dataset = TensorDataset(
    #         from_numpy(train_x),
    #         from_numpy(train_mean),
    #         from_numpy(train_var),
    #         from_numpy(train_y),
    #     )
    #     test_dataset = TensorDataset(
    #         from_numpy(test_x),
    #         from_numpy(test_mean),
    #         from_numpy(test_var),
    #         from_numpy(test_y),
    #     )
    #     return train_dataset, test_dataset


class SyntheticDatasetGetter(GluonTSDataManager):
    def __init__(self, name: str, multivariate: bool, generation_params) -> None:
        super().__init__(name, multivariate)
        self.generation_params = generation_params

    def get_main_dataset(self):
        pass
