import numpy as np

from gluonts.dataset import DataEntry
from gluonts.dataset.repository import get_dataset as gluonts_get_dataset
from gluonts.dataset.common import ListDataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper

TSDataset = list[np.ndarray]


class GluonTSDataManager:
    def __init__(self, name: str, multivariate: bool) -> None:
        self.name = name
        self.multivariate = multivariate
        self.init_main_dataset()
        # data from normalizer
        self.train_means = None
        self.train_vars = None
        self.test_means = None
        self.test_vars = None

    def init_main_dataset(self) -> None:
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
        gluonts_dataset = gluonts_get_dataset(self.name)
        self.n_features = len(list(gluonts_dataset.train)) if self.multivariate else 1
        assert gluonts_dataset.test is not None
        if self.multivariate:
            train_grouper = MultivariateGrouper(max_target_dim=self.n_features)
            test_grouper = MultivariateGrouper(
                max_target_dim=self.n_features,
                num_test_dates=len(list(gluonts_dataset.test)) // self.n_features,
            )
            self.train_dataset = train_grouper(gluonts_dataset.train)
            self.test_dataset = test_grouper(gluonts_dataset.test)
        else:
            self.train_dataset = gluonts_dataset.train
            self.test_dataset = gluonts_dataset.test

        assert isinstance(gluonts_dataset.metadata.prediction_length, int)
        self.prediction_length = gluonts_dataset.metadata.prediction_length
        self.context_length = 2 * self.prediction_length
        self.freq = gluonts_dataset.metadata.freq

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

    def _split_data_for_mean_layer(
        self, n_samples: int, phase: str, seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        assert phase in ["train", "test"], "Wrong phase"
        if phase == "train":
            dataset = [el["target"] for el in self.train_dataset]
            means = self.train_means
        else:
            dataset = [el["target"] for el in self.test_dataset]
            means = self.test_means
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
        ), "Data from normalizer not set"

        if not self.multivariate:
            train_dataset = ListDataset(
                [
                    {
                        "target": data_entry["target"],
                        "start": data_entry["start"],
                        "feat_dynamic_real": np.concatenate((mean, var), axis=1).T,
                    }
                    for data_entry, mean, var in zip(
                        self.train_dataset,
                        self.train_means,
                        self.train_vars,
                    )
                ],
                freq=self.freq,
            )
            test_dataset = ListDataset(
                [
                    {
                        "target": data_entry["target"],
                        "start": data_entry["start"],
                        "feat_dynamic_real": np.concatenate((mean, var), axis=1).T,
                    }
                    for data_entry, mean, var in zip(
                        self.test_dataset, self.test_means, self.test_vars
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
                        "feat_dynamic_real": np.concatenate((mean, var), axis=1).T,
                    }
                    for data_entry, mean, var in zip(
                        self.train_dataset,
                        self.train_means,
                        self.train_vars,
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
                        "feat_dynamic_real": np.concatenate((mean, var), axis=1).T,
                    }
                    for data_entry, mean, var in zip(
                        self.test_dataset, self.test_means, self.test_vars
                    )
                ],
                freq=self.freq,
                one_dim_target=False,
            )

        return train_dataset, test_dataset

    def get_torch_dataset_for_dl_layer(self):
        pass


class SyntheticDatasetGetter(GluonTSDataManager):
    def __init__(self, name: str, multivariate: bool, generation_params) -> None:
        super().__init__(name, multivariate)
        self.generation_params = generation_params

    def get_main_dataset(self):
        pass