import numpy as np
from scipy.optimize import minimize
from scipy.special import gamma
from tqdm import tqdm


class GASNormalizer:
    """
    GAS normalizer interface.
    """

    def __init__(self) -> None:
        pass

    def update_mean_and_var(
        self,
        ts_i: np.ndarray,
        mean: np.ndarray,
        var: np.ndarray,
        *args,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Method that computes the timestep update of the mean and variance.
        - ts_i, mean, var are 1D np.array of shape (n_features) and represent
          the current timestep of the time series, the mean and the variance.
        - args are the other static parameters of the normalizer.
        """
        raise NotImplementedError()

    def compute_neg_log_likelihood(
        self,
        ts: np.ndarray,
        mean_0: np.ndarray,
        var_0: np.ndarray,
        *args,
    ) -> float:
        """
        This methods compute the negative log likelihood of a single time series.
        - ts is a 2D np.ndarray of shape (ts_length, n_features)
        - mean_0 and var_0 are 1D np.ndarray of shape (n_features) and are the
          first values of mean and variance
        - args are the other static parameters of the normalizer.
        """
        raise NotImplementedError()

    @staticmethod
    def unpack_minimization_input(x: np.ndarray, n_features: int) -> dict:
        """
        This method unpacks the input of the minimization function that we want
        to minimize with scipy.optimize.minimize. This function takes as input a
        1D np.array. The length of this array depends on the number of (float) static
        parameters and initial values (usually only means and vars). Its length is
        (n_initial_values * n_features + n_static_parameters). The first n_features
        elements of the array is the first initial guess, the second n_features
        elements is the second initial guess and so on. About static parameters,
        the n_initial_guesses* n_features + 1 is the first one, etc.

        The method returns a dictionary {"param_name": param_value}, where param_name
        is the correct name in order to pass the variable to the update_mean_and_var
        and compute_neg_log_likelihood methods.
        """
        raise NotImplementedError()

    def warm_up(
        self,
        dataset: list[np.ndarray],
        initial_guesses: np.ndarray,
        bounds: tuple,
    ) -> list[dict]:
        """
        This method computes the ideal initial guesses and static parameters for
        each of the input time series in the list. Each time series is (len, n_feat).
        Ideal results are obtained as minimizers of the negative log likelihood.

        It returns the initial values and static parameters as lists dictionaries.

        Initial guesses must be a 1D array, look at unpack_minimization_input for
        the correct shapes description. Bounds must be an iterable of couples with
        the same length of initial_guesses.
        """

        n_features = dataset[0].shape[1]

        initial_params_list = []
        for ts in tqdm(dataset, total=len(dataset), unit="ts"):

            def func_to_minimize(x):
                # we must first unpack the input
                params = self.unpack_minimization_input(x, n_features)
                return self.compute_neg_log_likelihood(ts, **params)

            optimal = minimize(
                lambda x: func_to_minimize(x),
                x0=initial_guesses,
                bounds=bounds,
            )
            initial_params_list.append(
                self.unpack_minimization_input(optimal.x, n_features)
            )

        return initial_params_list

    def normalize(
        self, dataset: list[np.ndarray], normalizer_params: list[dict]
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """
        This method normalizes a dataset (list) of time series. It needs also time
        series parameters, which are the output of the warm_up method (i.e. ideal
        initial values and static parameters for each time series).

        It returns the dataset of normalized time series, their means and their
        vars. We will always assume 2D inputs time series (len, n_feat).
        """
        raise NotImplementedError()


class GASGaussian(GASNormalizer):
    """
    This class generalize GAS gaussian normalizer, i.e. they all use two static
    parameters (eta_mean and eta_var) and the same negative log likelihood function.
    """

    def __init__(self, eps: float = 1e-9) -> None:
        super(GASGaussian, self).__init__()
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
        for i, ts_i in enumerate(ts):
            mean, var = self.update_mean_and_var(ts_i, mean, var, eta_mean, eta_var)
            next_ts = ts[i + 1] if i != ts_length - 1 else ts_i
            log_likelihood_i = (
                -0.5 * np.log(2 * np.pi * var) - 0.5 * (next_ts - mean) ** 2 / var
            )
            neg_log_likelihood = neg_log_likelihood - log_likelihood_i

        return neg_log_likelihood / ts_length

    @staticmethod
    def unpack_minimization_input(x: np.ndarray, n_features: int) -> dict:
        """
        This method unpacks the input of the minimization function that we want
        to minimize with the SciPy method. It takes a 1D np.ndarray of shape
        (2 * n_features + 2, ) containing the initial guesses for:
        - means (n_features)
        - vars (n_features)
        - eta_mean (float)
        - eta_var (float)
        """
        result = {
            "mean_0": x[:n_features],
            "var_0": x[n_features : 2 * n_features],
            "eta_mean": x[2 * n_features],
            "eta_var": x[2 * n_features + 1],
        }
        return result

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
            eta_var = ts_params["eta_var"]  # float

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


class GASTStudent(GASNormalizer):
    def __init__(
        self, mean_strength: float, var_strength: float, eps: float = 1e-9
    ) -> None:
        super(GASTStudent, self).__init__()

        assert 0 <= mean_strength <= 0.5, "mean_strength must be between 0 and 0.5"
        assert 0 <= var_strength <= 0.5, "var_strength must be between 0 and 0.5"

        self.mean_strength = mean_strength
        self.var_strength = var_strength
        self.eps = eps

    def update_mean_and_var(
        self,
        ts_i: np.ndarray,
        mean: np.ndarray,
        var: np.ndarray,
        eta_mean: float,
        eta_var: float,
        nu: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        mean_updated = mean + (
            (self.mean_strength) / (1 - self.mean_strength)
        ) * eta_mean * (ts_i - mean) / (1 + (ts_i - mean) ** 2 / (nu * var))
        var_updated = var + (
            (self.var_strength) / (1 - self.var_strength)
        ) * eta_var * (
            (nu + 1) * (ts_i - mean) ** 2 / (nu + (ts_i - mean) ** 2 / var) - var
        )

        return mean_updated, var_updated

    def compute_neg_log_likelihood(
        self,
        ts: np.ndarray,
        mean_0: np.ndarray,
        var_0: np.ndarray,
        eta_mean: float,
        eta_var: float,
        nu: float,
    ) -> float:
        ts_length = ts.shape[0]
        neg_log_likelihood = 0

        mean, var = mean_0, var_0
        for i, ts_i in enumerate(ts):
            prev_mean, prev_var = mean, var
            mean, var = self.update_mean_and_var(ts_i, mean, var, eta_mean, eta_var, nu)
            penalty_term_mean = 0.5 * (1 - self.mean_strength) * (mean - prev_mean) ** 2
            penalty_term_var = 0.5 * (1 - self.var_strength) * (var - prev_var) ** 2

            next_ts = ts[i + 1] if i != ts_length - 1 else ts_i
            log_likelihood_i = (
                np.log(gamma((nu + 1) / 2))
                - np.log(gamma(nu / 2))
                - 0.5 * np.log(np.pi * nu)
                - 0.5 * np.log(var)
                - ((nu + 1) / 2) * np.log(1 + (next_ts - mean) ** 2 / (nu * var))
            )
            log_likelihood_i = (
                (self.mean_strength + self.var_strength) * log_likelihood_i
                - penalty_term_mean
                - penalty_term_var
            )

            neg_log_likelihood = neg_log_likelihood - log_likelihood_i
        return neg_log_likelihood / ts_length

    @staticmethod
    def unpack_minimization_input(x: np.ndarray, n_features: int) -> dict:
        """
        This method unpacks the input of the minimization function that we want
        to minimize with the SciPy method. It takes a 1D np.ndarray of shape
        (2 * n_features + 3, ) containing the initial guesses for:
        - means (n_features)
        - vars (n_features)
        - alpha_mean (float)
        - alpha_var (float)
        - nu (float)
        """
        result = {
            "mean_0": x[:n_features],
            "var_0": x[n_features : 2 * n_features],
            "eta_mean": x[2 * n_features],
            "eta_var": x[2 * n_features + 1],
            "nu": x[2 * n_features + 2],
        }
        return result

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
            eta_var = ts_params["eta_var"]  # float
            nu = ts_params["nu"]  # float

            for i, ts_i in enumerate(ts):
                mean, var = self.update_mean_and_var(
                    ts_i, mean, var, eta_mean, eta_var, nu
                )
                ts_means[i] = mean
                ts_vars[i] = var * nu / (nu - 2)  # check this
            norm_ts = (ts - ts_means) / (np.sqrt(ts_vars) + self.eps)

            norm_dataset.append(norm_ts)
            means.append(ts_means)
            vars.append(ts_vars)
        return norm_dataset, means, vars
