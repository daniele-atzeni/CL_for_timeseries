import numpy as np
from scipy.optimize import minimize
from scipy.special import gamma
from tqdm import tqdm

from joblib import Parallel, delayed


class GASNormalizer:
    """
    GAS normalizer interface.
    """

    def __init__(self) -> None:
        self.n_static_params = 0  # need this for the creation of the mean layer

    def update_mean_and_var(
        self,
        ts_i: np.ndarray | float,
        mean: np.ndarray | float,
        var: np.ndarray | float,
        *args,
    ) -> tuple[np.ndarray | float, np.ndarray | float]:
        """
        Method that computes the timestep update of the mean and variance.
        - ts_i, mean, var are 1D np.array of shape (n_features) and represent
          the current timestep of the time series, the mean and the variance.
        - args are the other static parameters of the normalizer.

        This method must also work for a single feature of a time series (during
        the minimization of the neg_log_likelihood), so inputs can also be float.
        """
        raise NotImplementedError()

    def compute_neg_log_likelihood(
        self,
        ts: np.ndarray,
        mean_0: float,
        var_0: float,
        *args,
    ) -> float:
        """
        This methods compute the negative log likelihood of a single time series
        feature. The inputs are:
        - ts is a 1D np.ndarray of shape (ts_length)
        - mean_0 and var_0 are float and are the first values of mean and variance
          time series that GAS outputs.
        - args are the other static parameters of the normalizer.
        """
        raise NotImplementedError()

    def warm_up(
        self,
        dataset: list[np.ndarray],
        context_length: int,
        initial_guesses: np.ndarray,
        bounds: tuple,
        use_context: bool = True,
    ) -> list:
        """
        This method computes the ideal initial guesses and static parameters for
        each (feature of) input time series in the list. Each time series is (len, n_feat).
        Ideal results are obtained as minimizers of the negative log likelihood.

        It returns the initial values and static parameters as list of numpy arrays
        of shape (n_feat * (2 + gas params)), where 2 is due to initial mean and var.

        Initial guesses must be a 1D array, look at unpack_minimization_input for
        the correct shapes description. Bounds must be an iterable of couples with
        the same length of initial_guesses.
        """

        n_features = dataset[0].shape[1]

        # the results will be a list of lists containing the optimal values  of
        # the parameters as numpy arrays (n_features,)
        initial_params_list = []
        verbose = 10
        if n_features == 1:

            def ts_inner_func(ts, use_context):
                ts_initial_guesses = initial_guesses.copy()
                if use_context:
                    mean_0 = np.mean(ts[:context_length], axis=0)
                    var_0 = np.var(ts[:context_length], axis=0)
                else:
                    mean_0 = np.mean(ts[:], axis=0)
                    var_0 = np.var(ts[:], axis=0)

                def func_to_minimize(x):
                    # we must first unpack the input
                    return self.compute_neg_log_likelihood(
                        ts.squeeze(), mean_0, var_0, *x
                    )

                optimal = minimize(
                    func_to_minimize, x0=ts_initial_guesses, bounds=bounds
                )
                # check if optimization worked
                # if not, set initial values as overall mean and var
                if not optimal.success and use_context:
                    return ts_inner_func(ts, use_context=False)

                # we need to add the values we compute before the optimization
                return np.concatenate([mean_0, var_0, optimal.x])

            results = Parallel(n_jobs=-1, verbose=verbose)(
                delayed(ts_inner_func)(ts, use_context) for ts in dataset
            )
            return list(results)
        else:
            for ts in tqdm(dataset, total=len(dataset), unit="ts"):
                # we normalize time_series features independently
                ts_results = []

                def feat_inner_func(feat, use_context):
                    # update initial guesses based on the time series
                    ts_initial_guesses = initial_guesses.copy()
                    if use_context:
                        mean_0 = np.mean(ts[:context_length, feat], axis=0)
                        var_0 = np.var(ts[:context_length, feat], axis=0)
                    else:
                        mean_0 = np.mean(ts[:, feat], axis=0)
                        var_0 = np.var(ts[:, feat], axis=0)

                    # we define the function to minimize
                    def func_to_minimize(x):
                        # we must first unpack the input
                        return self.compute_neg_log_likelihood(
                            ts[:, feat], mean_0, var_0, *x
                        )

                    optimal = minimize(
                        func_to_minimize,
                        x0=ts_initial_guesses,
                        bounds=bounds,
                    )
                    # check if optimization worked
                    # if not, set initial values as overall mean and var
                    if not optimal.success and use_context:
                        return feat_inner_func(feat, use_context=False)

                    return np.concatenate(np.array([mean_0, var_0]), optimal.x)

                # let's run the code for each feature in parallel
                ts_results = Parallel(n_jobs=-1, verbose=verbose)(
                    delayed(feat_inner_func)(feat, use_context) for feat in range(n_features)
                )  # this function mantains the ordering
                ts_results = np.stack(ts_results, axis=1)  # type: ignore I'm sure it's a list of np.ndarray

                initial_params_list.append([el for el in ts_results])
        return initial_params_list

    def normalize(
        self, dataset: list[np.ndarray], normalizer_params: list
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

        self.n_static_params = 2

    def update_mean_and_var(
        self,
        ts_i: np.ndarray | float,
        mean: np.ndarray | float,
        var: np.ndarray | float,
        eta_mean: np.ndarray | float,
        eta_var: np.ndarray | float,
    ) -> tuple[np.ndarray | float, np.ndarray | float]:
        raise NotImplementedError()

    def compute_neg_log_likelihood(
        self,
        ts: np.ndarray,
        mean_0: float,
        var_0: float,
        eta_mean: float,
        eta_var: float,
    ) -> float:
        """
        Method to compute negative log likelihood of a single time series feature.
        ts is assumed to be a 1D np.ndarray of shape (ts_length)
        mean_0 and var_0 are assumed to be float, as well as the other params.
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

    def normalize(
        self, dataset: list[np.ndarray], normalizer_params: list
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

            mean, var, eta_mean, eta_var = ts_params

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
        ts_i: np.ndarray | float,
        mean: np.ndarray | float,
        var: np.ndarray | float,
        eta_mean: np.ndarray | float,
        eta_var: np.ndarray | float,
    ) -> tuple[np.ndarray | float, np.ndarray | float]:
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
        ts_i: np.ndarray | float,
        mean: np.ndarray | float,
        var: np.ndarray | float,
        alpha_mean: np.ndarray | float,
        alpha_var: np.ndarray | float,
    ) -> tuple[np.ndarray | float, np.ndarray | float]:
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
        self,
        mean_strength: float,
        var_strength: float,
        eps: float = 1e-9,
    ) -> None:
        super(GASTStudent, self).__init__()

        assert 0 <= mean_strength <= 0.5, "mean_strength must be between 0 and 0.5"
        assert 0 <= var_strength <= 0.5, "var_strength must be between 0 and 0.5"

        self.mean_strength = mean_strength
        self.var_strength = var_strength
        self.eps = eps

        self.n_static_params = 7

    def update_mean_and_var(
        self,
        ts_i: np.ndarray | float,
        mean: np.ndarray | float,
        var: np.ndarray | float,
        alpha_mean: np.ndarray | float,
        alpha_var: np.ndarray | float,
        beta_mean: np.ndarray | float,
        beta_var: np.ndarray | float,
        omega_mean: np.ndarray | float,
        omega_var: np.ndarray | float,
        nu: np.ndarray | float,
    ) -> tuple[np.ndarray | float, np.ndarray | float]:
        mean_updated = mean + (
            (self.mean_strength) / (1 - self.mean_strength)
        ) * alpha_mean * (ts_i - mean) / (
            1 + (ts_i - mean) ** 2 / (nu * var + self.eps)
        )
        mean_updated = omega_mean + beta_mean * mean_updated

        var_updated = var + (
            (self.var_strength) / (1 - self.var_strength)
        ) * alpha_var * (
            (nu + 1) * (ts_i - mean) ** 2 / (nu + (ts_i - mean) ** 2 / (var + self.eps))
            - var
        )
        var_updated = omega_var + beta_var * var_updated

        return mean_updated, var_updated

    def compute_neg_log_likelihood(
        self,
        ts: np.ndarray,
        mean_0: float,
        var_0: float,
        alpha_mean: float,
        alpha_var: float,
        beta_mean: float,
        beta_var: float,
        omega_mean: float,
        omega_var: float,
        nu: float,
    ) -> float:
        ts_length = ts.shape[0]
        neg_log_likelihood = 0

        mean, var = mean_0, var_0
        for i, ts_i in enumerate(ts):
            prev_mean, prev_var = mean, var
            mean, var = self.update_mean_and_var(
                ts_i,
                mean,
                var,
                alpha_mean,
                alpha_var,
                beta_mean,
                beta_var,
                omega_mean,
                omega_var,
                nu,
            )
            penalty_term_mean = 0.5 * (1 - self.mean_strength) * (mean - prev_mean) ** 2
            penalty_term_var = 0.5 * (1 - self.var_strength) * (var - prev_var) ** 2

            next_ts = ts[i + 1] if i != ts_length - 1 else ts_i
            log_likelihood_i = (
                np.log(gamma((nu + 1) / 2))
                - np.log(gamma(nu / 2))
                - 0.5 * np.log(np.pi * nu)
                - 0.5 * np.log(var + self.eps)
                - ((nu + 1) / 2)
                * np.log(1 + (next_ts - mean) ** 2 / (nu * var + self.eps))
            )
            log_likelihood_i = (
                (self.mean_strength + self.var_strength) * log_likelihood_i
                - penalty_term_mean
                - penalty_term_var
            )

            neg_log_likelihood = neg_log_likelihood - log_likelihood_i
        return neg_log_likelihood / ts_length

    def normalize(
        self, dataset: list[np.ndarray], normalizer_params: list[np.ndarray]
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

            (
                mean,
                var,
                alpha_mean,
                alpha_var,
                beta_mean,
                beta_var,
                omega_mean,
                omega_var,
                nu,
            ) = ts_params

            for i, ts_i in enumerate(ts):
                mean, var = self.update_mean_and_var(
                    ts_i,
                    mean,
                    var,
                    alpha_mean,
                    alpha_var,
                    beta_mean,
                    beta_var,
                    omega_mean,
                    omega_var,
                    nu,
                )
                ts_means[i] = mean
                ts_vars[i] = var * nu / (nu - 2)  # check this
            norm_ts = (ts - ts_means) / (np.sqrt(ts_vars) + self.eps)

            norm_dataset.append(norm_ts)
            means.append(ts_means)
            vars.append(ts_vars)
        return norm_dataset, means, vars
