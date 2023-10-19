import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
from scipy.optimize import minimize

TsElement = float | np.ndarray | Tensor
Ts = np.ndarray | Tensor
TsDataset = list[Ts]


class GASNormalizer(nn.Module):
    """
    Define the GASNormalizer interface. This interface assumes that data is composed
    of a list of uni/multi-variate time series.

    GASNormalizer are stateful, because we need to compute the means and variances
    of the complete time series. So, we need to call a warm up method to compute them.

    The forward pass of the Normalizer takes as input
    - a batch of time series windows of a fixed length (i.e., tensors of shape
      (batch, context_length) for univariate and (batch, context_length, n_features)
      for multivariate)
    - tensor of indices of windows' original time series in the dataset list (shape (batch))
    - the indices of the time series windows in the original time series (i.e., tensors
      of shape (batch, context_length) containing integers)

    """

    def __init__(
        self,
        eps: float = 1e-9,
        mean_0: TsElement | None = None,
        var_0: TsElement | None = None,
    ) -> None:
        super(GASNormalizer, self).__init__()
        self.means = []
        self.vars = []
        self.mean_0 = mean_0
        self.var_0 = var_0
        self.eps = eps

    def has_means_and_vars(self) -> bool:
        if len(self.means) > 0 and len(self.vars) == 0:
            raise ValueError(
                "Something went wrong in the warm-up, means are present but vars are not."
            )
        if len(self.means) == 0 and len(self.vars) > 0:
            raise ValueError(
                "Something went wrong in the warm-up, vars are present but means are not."
            )
        return len(self.means) > 0 and len(self.vars) > 0

    def get_means_and_vars(self) -> tuple[TsDataset, TsDataset]:
        if not self.has_means_and_vars():
            raise ValueError(
                "You must call the warm_up method before using the normalizer."
            )
        return self.means, self.vars

    def compute_static_parameters(self, ts: TsDataset) -> None:
        raise NotImplementedError()

    def update_mean_and_var(  # this wants only tensors
        self,
        ts_i: Tensor,
        mean: Tensor,
        var: Tensor,
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError()

    def compute_single_ts_means_and_vars(self, ts: Ts) -> tuple[Ts, Ts]:
        # ts is the complete time series (shape = (total_length, n_features))
        # we want to compute mus and vars of the same shape
        # intialize the results
        is_numpy = isinstance(ts, np.ndarray)
        if is_numpy:
            ts = torch.from_numpy(ts)

        means = torch.empty_like(ts)
        vars = torch.empty_like(ts)
        # intialize first mean and var
        mean = torch.tensor(self.mean_0) if self.mean_0 else torch.mean(ts, dim=0)
        var = torch.tensor(self.var_0) if self.var_0 else torch.var(ts, dim=0)
        # compute mus and vars
        for i, ts_i in enumerate(ts):
            mean, var = self.update_mean_and_var(ts_i, mean, var)
            means[i] = mean
            vars[i] = var

        if is_numpy:
            means = means.numpy()
            vars = vars.numpy()

        return means, vars

    def compute_means_and_vars(self, ts: TsDataset) -> None:
        """
        Compute means and variances of the complete dataset. Remember that
        dataset is a list of time series, so ts is a list of array/tensors. Note also that
        each time series can be univariate or multivariate, so each tensor can be
        (ts_length) or (ts_length, n_features). We must always pass 2D tensors to
        compute_single_ts_means_and_vars, so we must check the shape of each tensor
        """
        for ts_i in ts:
            if len(ts_i.shape) == 1:
                if isinstance(ts_i, np.ndarray):
                    ts_i = np.expand_dims(ts_i, axis=1)
                else:
                    ts_i = ts_i.unsqueeze(1)
            means_i, vars_i = self.compute_single_ts_means_and_vars(ts_i)
            self.means.append(means_i)
            self.vars.append(vars_i)
        return

    def warm_up(
        self,
        train_ts: TsDataset,
        complete_ts: TsDataset,
    ) -> None:
        """
        This method compute the ideal static parameters of the normalizer given
        the training time series, then precomputes means and variances of the
        complete time series. In GluonTS, the input is a list of numpy array, with
        PyTorch is a tensor (len_ts, n_features)
        """
        if self.has_means_and_vars():
            raise ValueError(
                "You must call the warm_up method only once before using the normalizer."
            )
        self.compute_static_parameters(train_ts)
        self.compute_means_and_vars(complete_ts)
        return

    def normalize(
        self,
        ts: Tensor | TsDataset,
        means: Tensor | None = None,
        vars: Tensor | None = None,
    ) -> list[np.ndarray] | Tensor:
        """
        This method is suppose to implement normalization equation. So it must be called
        for in the forward pass of the PyTorch training loop, or normalize the whole
        time series in a GluonTS framework (as a list of numpy array). In the second case,
        we do not expect means and vars because we are using the saved ones in the
        warm up phase
        """
        if isinstance(ts, Tensor):
            assert (
                means is not None and vars is not None
            ), "You must pass means and vars"
            return (ts - means) / (torch.sqrt(vars) + self.eps)
        else:
            # GluonTS case, we will use self.means and self.vars
            # remember that these can be longer of the input time series
            # indeed, they are as long as the complete time series (test one)
            assert means is None and vars is None, "You must not pass means and vars"
            # remember means and vars contain 2D tensors
            squeezed_means = [m.squeeze(1) for m in self.means]
            squeezed_vars = [v.squeeze(1) for v in self.vars]
            normalized_ts = []
            for ts_i, mean_i, var_i in zip(ts, squeezed_means, squeezed_vars):
                ts_len = ts_i.shape[0]
                normalized_ts.append(
                    (ts_i - mean_i[:ts_len]) / (np.sqrt(var_i[:ts_len]) + self.eps)
                )
            return normalized_ts

    def forward(
        self, ts_index: Tensor, window_indices: Tensor, ts: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        This method is used in the forward part of a PyTorch training loop. It normalizes
        batches of the time series window by retrieving the means and variances
        - ts_index is a Tensor (batch) of int with the index of the windows' original time series
          in the dataset list
        - window_indices are the indices of the time series windows in the original time series
          (tensor of shape (batch, context_length) containing integers)
        - ts is the complete time series (shape (batch, window_length, n_features))
        """
        if not self.has_means_and_vars():
            raise ValueError(
                "You must call the warm_up method before using the normalizer."
            )
        # check the shape, if each window is 1D add a dimension to have the same shape of means and vars
        if ts.dim() == 2:
            ts = ts.unsqueeze(2)
        # retrieve means and var
        means, vars = torch.empty_like(ts), torch.empty_like(ts)
        for i, (ts_ind, windows_ind) in enumerate(zip(ts_index, window_indices)):
            means[i] = self.means[ts_ind][windows_ind]
            vars[i] = self.vars[ts_ind][windows_ind]
        # normalize
        normalized_window = self.normalize(ts, means, vars)
        assert isinstance(
            normalized_window, Tensor
        ), "Something went wrong with normalization"
        # return the same shape of the input
        if ts.dim() == 2:
            normalized_window = normalized_window.squeeze(2)

        return normalized_window, means, vars


class GASSimpleGaussian(GASNormalizer):
    def __init__(self, eta_mean: float, eta_var: float, eps: float = 1e-9) -> None:
        super(GASSimpleGaussian, self).__init__(eps)
        self.eta_mean = eta_mean
        self.eta_var = eta_var

    def compute_static_parameters(self, train_ts: Tensor) -> None:
        assert (
            self.eta_mean is not None and self.eta_var is not None
        ), "eta_mu and eta_var should be set by the initializer"
        return

    def update_mean_and_var(
        self, ts_i: Tensor, mean: Tensor | float, var: Tensor | float
    ) -> tuple[Tensor | float, Tensor | float]:
        """
        Method to compute means and vars of a single time step of a given (multivariate) time series
        """
        if self.eta_var is None or self.eta_mean is None:
            raise ValueError(
                "Something went wrong with the warm up, static parameters are not initialized."
            )
        # ts_i is the time series single element (shape = (n_features))
        # same shape for mu and var
        mean = mean + self.eta_mean * (ts_i - mean)
        var = var * (1 - self.eta_var) + self.eta_var * (ts_i - mean) ** 2
        return mean, var


class GASComplexGaussian(GASNormalizer):
    def __init__(
        self,
        initial_guesses: np.ndarray = np.array([0.001, 0.001, 0, 1]),  # type: ignore
        regularization: str = "full",
        eps: float = 1e-9,
    ) -> None:
        super(GASComplexGaussian, self).__init__(eps)
        # self.alpha_mu, self.alpha_sigma, self.mean_0, self.var_0 = initial_guesses
        self.initial_guesses = initial_guesses
        self.regularization = regularization

        self.alpha_mean, self.alpha_sigma, self.mean_0, self.sigma2_0 = initial_guesses

    def update_mean_and_var(
        self,
        ts_i: float | Tensor | np.ndarray,
        mean: float | Tensor | np.ndarray,
        var: float | Tensor | np.ndarray,
        alpha_mean: float | None = None,
        alpha_sigma: float | None = None,
    ):
        if alpha_mean is None:
            alpha_mean = self.alpha_mean
        if alpha_sigma is None:
            alpha_sigma = self.alpha_sigma

        assert (
            alpha_mean is not None and alpha_sigma is not None
        ), "alpha_mean and alpha_sigma should be set by the initializer"

        if self.regularization == "full":
            mean_updated = mean + alpha_mean * (ts_i - mean)
            var_updated = var + alpha_sigma * ((ts_i - mean) ** 2 - var)
        elif self.regularization == "root":
            mean_updated = alpha_mean * (ts_i - mean) / (np.sqrt(var) + self.eps) + mean
            var_updated = (
                alpha_sigma
                * (
                    -np.sqrt(2) / 2
                    + np.sqrt(2) * (ts_i - mean) ** 2 / (2 * var + self.eps)
                )
                + var
            )
        else:
            raise ValueError("Error: regularization must be Full or Root")
        return mean_updated, var_updated

    def neg_log_likelihood_Gaussian_single_ts(
        self,
        ts: np.ndarray | Tensor,
        mean_0: float | Tensor | np.ndarray,
        var_0: float | Tensor | np.ndarray,
        alpha_mean: float,
        alpha_sigma: float,
    ) -> float:
        T = ts.shape[0]
        mean_ts, var_ts = np.zeros(T), np.zeros(T)
        log_likelihood_ts = np.zeros(T)
        """y = np.append(y, y[T - 1])"""

        mean, var = mean_0, var_0
        for i, ts_i in enumerate(ts):
            mean, var = self.update_mean_and_var(
                ts_i, mean, var, alpha_mean, alpha_sigma
            )
            mean_ts[i] = mean
            var_ts[i] = var
            log_likelihood_ts[i] = -0.5 * np.log(2 * np.pi * var + self.eps) - 0.5 * (
                ts_i - mean
            ) ** 2 / (var + self.eps)

        neg_log_lokelihood = -np.sum(log_likelihood_ts)
        return float(neg_log_lokelihood / T)

    def neg_log_likelihood_Gaussian(
        self,
        list_ts: list[np.ndarray | Tensor],
        mean_0: float,
        var_0: float,
        alpha_mean: float,
        alpha_sigma: float,
    ) -> float:
        return np.sum(
            [
                self.neg_log_likelihood_Gaussian_single_ts(
                    ts, mean_0, var_0, alpha_mean, alpha_sigma
                )
                for ts in list_ts
            ]
        )

    def compute_static_parameters(self, ts: list[np.ndarray | Tensor]) -> None:
        bounds = ((0, 1), (0, 1), (None, None), (0.00001, 1))

        def minimization_funct(ts, alpha_mean, alpha_sigma, mean_0, sigma2_0):
            return self.neg_log_likelihood_Gaussian(
                ts, alpha_mean, alpha_sigma, mean_0, sigma2_0
            )

        optimal = minimize(
            lambda params: minimization_funct(ts, *params),
            x0=self.initial_guesses,
            bounds=bounds,
        )

        self.alpha_mean, self.alpha_sigma, self.mean_0, self.sigma2_0 = optimal.x
        print(
            "Optimal parameters:  alpha_mean = {},  alpha_sigma = {}, mean_0 = {}, sigma2_0 = {}".format(
                self.alpha_mean, self.alpha_sigma, self.mean_0, self.sigma2_0
            )
        )
        return
