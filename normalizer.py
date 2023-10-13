import torch
from torch import Tensor
import torch.nn as nn
import numpy as np


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

    def __init__(self, eps: float = 1e-9) -> None:
        super(GASNormalizer, self).__init__()
        self.means = []
        self.vars = []
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

    def get_means_and_vars(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if not self.has_means_and_vars():
            raise ValueError(
                "You must call the warm_up method before using the normalizer."
            )
        return self.means, self.vars

    def compute_static_parameters(self, ts: list[np.ndarray | Tensor]) -> None:
        raise NotImplementedError()

    def compute_single_ts_means_and_vars(
        self, ts: np.ndarray | Tensor
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError()

    def compute_means_and_vars(self, ts: list[np.ndarray | Tensor]) -> None:
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
        train_ts: list[np.ndarray | Tensor],
        complete_ts: list[np.ndarray | Tensor],
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
        ts: Tensor | list[np.ndarray],
        means: Tensor | None = None,
        vars: Tensor | None = None,
    ) -> list[np.ndarray] | Tensor:
        """
        This method is suppose to implement normalization equation. So it must be called
        for normalize the whole time series in a GluonTS framework (as a list of numpy array),
        or in the forward pass of the PyTorch training loop. In the first case,
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

    def compute_single_ts_means_and_vars(
        self, ts: Tensor | np.ndarray
    ) -> tuple[Tensor | np.ndarray, Tensor | np.ndarray]:
        # ts is the complete time series (shape = (total_length, n_features))
        # we want to compute mus and vars of the same shape
        # intialize the results
        is_numpy = isinstance(ts, np.ndarray)
        if is_numpy:
            ts = torch.from_numpy(ts)

        means = torch.empty_like(ts)
        vars = torch.empty_like(ts)
        # intialize first mean and var
        mean = torch.mean(ts, dim=0)
        var = torch.var(ts, dim=0)
        means[0] = mean
        vars[0] = var
        # compute mus and vars
        for i, ts_i in enumerate(ts):
            mean, var = self.update_mean_and_var(ts_i, mean, var)
            means[i] = mean
            vars[i] = var

        if is_numpy:
            means = means.numpy()
            vars = vars.numpy()

        return means, vars
