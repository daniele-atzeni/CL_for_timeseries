import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm


class GASGaussian:
    def __init__(self, eps: float = 1e-9) -> None:
        self.eps = eps

    def update_mean_and_var(
        self,
        ts_i: np.ndarray,
        mean: np.ndarray,
        var: np.ndarray,
        eta_mean: float,
        eta_var: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def compute_neg_log_likelihood(
        self,
        ts: np.ndarray,
        mean_0: np.ndarray,
        var_0: np.ndarray,
        eta_mean: float,
        eta_var: float,
    ) -> float:
        """
        Method to compute negative log likelihood of a single time series.
        ts is assumed to be a 2D np.ndarray of shape (ts_length, n_features)
        mean_0 and var_0 are assumed to be 1D np.ndarray of shape (n_features)
        """

        ts_length = ts.shape[0]
        neg_log_likelihood = 0

        mean, var = mean_0, var_0
        for ts_i in ts:
            mean, var = self.update_mean_and_var(ts_i, mean, var, eta_mean, eta_var)
            neg_log_likelihood = (
                neg_log_likelihood
                - np.sum(  # we must sum because mean and var are arraysS
                    -0.5 * np.log(2 * np.pi * var + self.eps)
                    - 0.5 * (ts_i - mean) ** 2 / (var + self.eps)
                )
            )
        return neg_log_likelihood / ts_length

    @staticmethod
    def unpack_minimization_input(
        x: np.ndarray, n_features: int
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """
        This method unpacks the input of the minimization function that we want
        to minimize with the SciPy method. It takes a 1D np.ndarray of shape
        (2 * n_features + 2, ) containing the initial guesses for:
        - means (n_features)
        - vars (n_features)
        - alpha_mean (float)
        - alpha_var (float)
        """
        mean_0 = x[:n_features]
        var_0 = x[n_features : 2 * n_features]
        eta_mean = x[2 * n_features]
        eta_var = x[2 * n_features + 1]
        return mean_0, var_0, eta_mean, eta_var

    def warm_up(
        self,
        dataset: list[np.ndarray],
        initial_guesses: np.ndarray = np.array([0.001, 0.001, 0, 1], dtype="float"),
        bounds: tuple = ((None, None), (0.00001, 1), (0, 0.999), (0, 0.999)),
    ) -> list[dict]:
        """
        This method computes the ideal initial guesses and static parameters for
        each of the input time series in the list. Each time series is (len, n_feat)
        Ideal results are obtained as minimizers of the negative log likelihood function.
        It returns the initial guesses and static parameters as lists of numpy arrays.
        For GASGaussian models, we have 4 parameters:
        - mean_0: array of shape (n_features)
        - var_0: array of shape (n_features)
        - eta_mean: float
        - eta_var: float
        So, initial guesses must be a 1D array of shape (2 * n_features + 2, ).
        First n_features elements are the means of each feature, second n_features
        elements are the vars of each feature, last two elements are eta_mean and
        eta_var.
        Bounds must be an iterable with two bounds for each of the elements
        in initial guesses.
        """

        n_features = dataset[0].shape[1]
        # let's check that initial guesses and bounds are of the correct shape
        assert (
            initial_guesses.shape[0] == 2 * n_features + 2
        ), "initial_guesses must be a 1D array of shape (2 * n_features + 2, ). First n_features elements are the means of each feature, second n_features elements are the vars of each feature, last two elements are alpha_mean and alpha_var"
        assert (
            len(bounds) == 2 * n_features + 2
        ), "bounds must be a tuple of 2 * n_features + 2 elements (pair of values). First n_features elements are bounds for means of each feature, second n_features elements are bounds for vars of each feature, last two elements are bounds for alpha_mean and alpha_var"

        initial_params_list = []
        for ts in tqdm(dataset, total=len(dataset), unit="ts"):
            # define minimization function. It takes a 1D np.ndarray as input
            # the shape of this input is (4 * n_features, ) containing the initial
            # guesses for each ts feature and for each of the 4 parameter of the
            # normalizer
            def func_to_minimize(x):
                # we must first unpack the input
                mean_0, var_0, alpha_mean, alpha_var = self.unpack_minimization_input(
                    x, n_features
                )
                return self.compute_neg_log_likelihood(
                    ts, mean_0, var_0, alpha_mean, alpha_var
                )

            optimal = minimize(
                lambda x: func_to_minimize(x),
                x0=initial_guesses,
                bounds=bounds,
            )
            mean_0, var_0, eta_mean, eta_sigma = self.unpack_minimization_input(
                optimal.x, n_features
            )

            initial_params_list.append(
                {
                    "eta_mean": eta_mean,
                    "eta_sigma": eta_sigma,
                    "mean_0": mean_0,
                    "var_0": var_0,
                }
            )
        return initial_params_list

    def normalize(
        self, dataset: list[np.ndarray], normalizer_params: list[dict]
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """
        This method normalizes a dataset (list) of time series. It needs also time
        series parameters, which are the output of the warm_up method. It returns
        the dataset of normalized time series, the means and the vars of each time
        series. We will always assume 2D time series (len, n_feat)
        """
        norm_dataset, means, vars = [], [], []
        for ts, ts_params in zip(dataset, normalizer_params):
            ts_means = np.empty_like(ts)
            ts_vars = np.empty_like(ts)

            mean = ts_params["mean_0"]  # (n_features)
            var = ts_params["var_0"]  # (n_features)
            eta_mean = ts_params["eta_mean"]  # float
            eta_var = ts_params["eta_sigma"]  # float

            for i, ts_i in enumerate(ts):
                mean, var = self.update_mean_and_var(ts_i, mean, var, eta_mean, eta_var)
                ts_means[i] = mean
                ts_vars[i] = var
            norm_ts = (ts - ts_means) / (np.sqrt(ts_vars) + self.eps)

            norm_dataset.append(norm_ts)
            means.append(ts_means)
            vars.append(ts_vars)
        return norm_dataset, means, vars


class GASSimpleGaussian(GASGaussian):
    def __init__(self, eps: float = 1e-9) -> None:
        super(GASSimpleGaussian, self).__init__(eps=eps)

    def update_mean_and_var(
        self,
        ts_i: np.ndarray,
        mean: np.ndarray,
        var: np.ndarray,
        eta_mean: float,
        eta_var: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        mean_updated = mean + eta_mean * (ts_i - mean)
        var_updated = var * (1 - eta_var) + eta_var * (ts_i - mean) ** 2

        return mean_updated, var_updated


class GASComplexGaussian(GASGaussian):
    def __init__(
        self,
        eps: float = 1e-9,
        regularization: str = "full",
    ) -> None:
        super(GASComplexGaussian, self).__init__(eps=eps)
        self.regularization = regularization

    def update_mean_and_var(
        self,
        ts_i: np.ndarray,
        mean: np.ndarray,
        var: np.ndarray,
        alpha_mean: float,
        alpha_var: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Method to compute the single element update. Every input array is supposed
        to be a 1D np.ndarray of shape (n_features)
        """
        if self.regularization == "full":
            mean_updated = mean + alpha_mean * (ts_i - mean)
            var_updated = var + alpha_var * ((ts_i - mean) ** 2 - var)

        elif self.regularization == "root":
            mean_updated = alpha_mean * (ts_i - mean) / (np.sqrt(var) + self.eps) + mean
            var_updated = (
                alpha_var
                * (
                    -np.sqrt(2) / 2
                    + np.sqrt(2) * (ts_i - mean) ** 2 / (2 * var + self.eps)
                )
                + var
            )
        else:
            raise ValueError("Error: regularization must be 'full' or 'root'")
        return mean_updated, var_updated
