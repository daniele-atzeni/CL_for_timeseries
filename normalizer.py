import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

TsElement = float | np.ndarray | Tensor
Ts = np.ndarray | Tensor
TsDataset = list[Ts]


class GASNormalizer(nn.Module):
    """
    Define the GASNormalizer interface. This interface assumes that data is composed
    of a list of uni/multi-variate time series.

    GASNormalizer are stateful, because we need to compute the means and variances
    of the complete time series. So, we need to call a warm up method to compute them.
    Means and variances will be saved as lists of 2D tensors (shape = (ts_length, n_features))
    even in the univariate case.

    The forward pass of the Normalizer is needed only for the integration with
    PyTorch Modules, and it takes as input
    - time series indices (i.e., a tensor of shape (batch) containing integers),
      that correspond to the indices of the time series in the dataset list
    - window indices (i.e., tensors of shape (batch, context_length) containing integers),
      that correspond to the indices of the time series windows in the original time series
    - the values of the time series windows (i.e., tensors of shape (batch, context_length)
      for univariate and (batch, context_length, n_features) for multivariate)
    """

    def __init__(self, eps: float = 1e-9) -> None:
        super(GASNormalizer, self).__init__()
        # REMEMBER, a dataset is a list of time series

        # this class will pre-compute means and vars, which are the same type of the dataset
        self.means = []
        self.vars = []
        # initial values for the first mean and var in used to compute normalized time series
        # are the same type of time series elements (number or array)
        # this two values are initialized in the warm_up method
        self.means_0 = []
        self.vars_0 = []
        # epsilon to avoid division by zero
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
        """
        this method is used to compute the static parameters of the normalizer
        it depends on the implementation of the instance of GAS normalizer
        """
        raise NotImplementedError()

    def update_mean_and_var(
        self,
        ts_i: Tensor,
        mean: Tensor,
        var: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        this method works only with PyTorch tensors, even if its behavior can be
        also implemented with general arrays (eg numpy)
        generally speaking, it expects its input as 1D tensors (shape = (n_features))
        """
        raise NotImplementedError()

    def compute_single_ts_means_and_vars(
        self, ts: Ts, mean_0: Tensor, var_0: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        This method gets a single time series, converts its elements to tensors
        and compute the means and vars of the time series. It returns the means and
        vars as PyTorch Tensors. The accepted inputs are:
        - numpy array (shape = (ts_length, n_features))
        - PyTorch tensor (shape = (ts_length, n_features))
        """
        # ts is the complete time series (shape = (total_length, n_features))
        # we want to compute mus and vars of the same shape
        # intialize the results
        if isinstance(ts, np.ndarray):
            ts = torch.from_numpy(ts)

        means = torch.empty_like(ts)
        vars = torch.empty_like(ts)
        # intialize first mean and var
        mean = mean_0
        var = var_0
        # compute mus and vars
        for i, ts_i in enumerate(ts):
            mean, var = self.update_mean_and_var(ts_i, mean, var)
            means[i] = mean
            vars[i] = var

        return means, vars

    def compute_means_and_vars(self, ts: TsDataset) -> None:
        """
        Compute means and variances of the complete dataset. Remember that
        dataset is a list of time series. Note also that each time series can be
        univariate or multivariate, so each tensor can be (ts_length) or
        (ts_length, n_features). We must always pass 2D tensors to
        compute_single_ts_means_and_vars, so we must check the shape of each tensor
        """
        for ts_i, mean_0, var_0 in zip(ts, self.means_0, self.vars_0):
            if len(ts_i.shape) == 1:
                if isinstance(ts_i, np.ndarray):
                    ts_i = np.expand_dims(ts_i, axis=1)
                else:
                    ts_i = ts_i.unsqueeze(1)
            means_i, vars_i = self.compute_single_ts_means_and_vars(ts_i, mean_0, var_0)
            self.means.append(means_i)
            self.vars.append(vars_i)
        return

    def warm_up(
        self,
        train_dataset: TsDataset,
        complete_dataset: TsDataset,
    ) -> None:
        """
        This method compute the ideal static parameters of the normalizer given
        the training time series, then precomputes means and variances of the
        complete time series as lists of 2D torch tensors.
        """
        if self.has_means_and_vars():
            raise ValueError(
                "You must call the warm_up method only once before using the normalizer."
            )
        self.compute_static_parameters(train_dataset)

        # we must check if mean_0 and var_0 are initialized
        # if not, we initialize them as mean and var of the time series
        # they must have the same shape of means and vars, i.e., lists of 2D tensors
        if len(self.means_0) == 0:
            self.means_0 = [
                torch.mean(torch.Tensor(ts_i), dim=0) for ts_i in complete_dataset
            ]
        if len(self.vars_0) == 0:
            self.vars_0 = [
                torch.var(torch.Tensor(ts_i), dim=0) for ts_i in complete_dataset
            ]
        # each element of this list must be a tensor (n_features)
        self.means_0 = [
            mean_0.reshape(-1) for mean_0 in self.means_0 if mean_0.dim() == 0
        ]
        self.vars_0 = [var_0.reshape(-1) for var_0 in self.vars_0 if var_0.dim() == 0]

        self.compute_means_and_vars(complete_dataset)
        return

    def normalize(
        self,
        ts: Tensor | TsDataset,
        means: Tensor | None = None,
        vars: Tensor | None = None,
    ) -> Tensor | TsDataset:
        """
        This method is suppose to implement the normalization equation. So it must
        be called in two situations:

        - in the forward pass of the PyTorch training loop, when we 1) have the means and
          vars of the time series windows, and 2) ts is a tensor. All the tensor in
          this case are (batch, context_length, n_features) even in univariate case

        - to normalize a whole dataset of time series. In this case, we do not expect
          means and vars, because we are using the saved ones in the warm up phase.
          Remember to cast arrays to tensors, because means and vars are saved as tensors.
          Also, remember that means and vars are lists of 2D tensors, so we must
          convert possibly univariate time series in 2D tensors.

        The returned value has the same type of the input ts.
        """
        if isinstance(ts, Tensor):
            assert (
                means is not None and vars is not None
            ), "You must pass means and vars"
            return (ts - means) / (torch.sqrt(vars) + self.eps)
        else:  # ts is a list of time series
            assert means is None and vars is None, "You must not pass means and vars"
            # remember means and vars contain 2D tensors
            is_univariate = len(ts[0].shape) == 1
            is_numpy = isinstance(ts[0], np.ndarray)
            ts_tensor = [torch.from_numpy(ts_i) for ts_i in ts if is_numpy]
            if is_univariate:
                ts_tensor = [ts_i.unsqueeze(1) for ts_i in ts_tensor]
            normalized_ts = []
            for ts_i, mean_i, var_i in zip(ts_tensor, self.means, self.vars):
                ts_len = ts_i.shape[0]
                # we will take only the first ts_len elements, because means and
                # vars are as long as the test time series
                normalized_ts.append(
                    (ts_i - mean_i[:ts_len]) / (np.sqrt(var_i[:ts_len]) + self.eps)
                )
            # we must return the same type of the input
            if is_univariate:
                normalized_ts = [ts_i.squeeze(1) for ts_i in normalized_ts]
            if is_numpy:
                normalized_ts = [ts_i.numpy() for ts_i in normalized_ts]
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


"""
Ideally, the only thing that should change from the Base class of the GAS normalizer are:

- the initialization, in which we must include the static parameters of the normalizer.
  Remember that if we want, we can also pass eps, mean_0 and var_0, but they are not necessary.

- the compute_static_parameters method, in which we must compute the static parameters
  of the normalizer. This method is supposed to work with a list of PyTorch tensors
  or numpy arrays of shapes either (ts_length, n_features) or (ts_length).

- the update_mean_and_var method, in which we must update the mean and var of a single
  time step of the time series, given the previous mean, the  previous 
  var, and the current value of the time series. Inputs are tensors of the same shape.
"""


class GASSimpleGaussian(GASNormalizer):
    def __init__(self, eta_mean: float, eta_var: float, **kwargs) -> None:
        super(GASSimpleGaussian, self).__init__(**kwargs)
        self.eta_mean = eta_mean
        self.eta_var = eta_var

    def compute_static_parameters(self, ts: TsDataset) -> None:
        # we don't need to compute static parameters in this case
        assert (
            self.eta_mean is not None and self.eta_var is not None
        ), "eta_mu and eta_var should be set by the initializer"
        return

    def update_mean_and_var(
        self, ts_i: Tensor, mean: Tensor, var: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Method to compute means and vars of a single time step of a given (multivariate) time series
        """
        if self.eta_var is None or self.eta_mean is None:
            raise ValueError(
                "Something went wrong with the warm up, static parameters are not initialized."
            )
        mean = mean + self.eta_mean * (ts_i - mean)
        var = var * (1 - self.eta_var) + self.eta_var * (ts_i - mean) ** 2
        return mean, var


'''
class GASComplexGaussian(GASNormalizer):
    def __init__(
        self,
        regularization: str = "full",
        **kwargs,
    ) -> None:
        super(GASComplexGaussian, self).__init__(**kwargs)

        self.alpha_means = []
        self.alpha_vars = []

        self.regularization = regularization

    def update_mean_and_var(
        self,
        ts_i: Tensor,
        mean: Tensor,
        var: Tensor,
        alpha_mean: Tensor,
        alpha_var: Tensor,
    ):
        """
        In this case, we must take as input also alpha_mean and alpha_var, because
        to use sklearn minimize function, we must have a function that takes as input
        a single vector of parameters. Since the negative log likelihood uses this
        method, we must allow these parameters to be passed as input.

        In the forward pass, we will use the optimal values of alpha_mean and alpha_var.
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

    def compute_single_ts_means_and_vars(
        self,
        ts: Ts,
        mean_0: Tensor,
        var_0: Tensor,
        alpha_mean: Tensor,
        alpha_var: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        This method gets a single time series, converts its elements to tensors
        and compute the means and vars of the time series. It returns the means and
        vars as PyTorch Tensors. The accepted inputs are:
        - numpy array (shape = (ts_length, n_features))
        - PyTorch tensor (shape = (ts_length, n_features))
        """
        # ts is the complete time series (shape = (total_length, n_features))
        # we want to compute mus and vars of the same shape
        # intialize the results
        if isinstance(ts, np.ndarray):
            ts = torch.from_numpy(ts)

        means = torch.empty_like(ts)
        vars = torch.empty_like(ts)
        # intialize first mean and var
        mean = mean_0
        var = var_0
        # compute mus and vars
        for i, ts_i in enumerate(ts):
            mean, var = self.update_mean_and_var(ts_i, mean, var, alpha_mean, alpha_var)
            means[i] = mean
            vars[i] = var

        return means, vars

    def compute_means_and_vars(self, ts: TsDataset) -> None:
        """
        Compute means and variances of the complete dataset. Remember that
        dataset is a list of time series. Note also that each time series can be
        univariate or multivariate, so each tensor can be (ts_length) or
        (ts_length, n_features). We must always pass 2D tensors to
        compute_single_ts_means_and_vars, so we must check the shape of each tensor
        """
        for ts_i, mean_0, var_0, alpha_mean, alpha_var in zip(
            ts, self.means_0, self.vars_0, self.alpha_means, self.alpha_vars
        ):
            if len(ts_i.shape) == 1:
                if isinstance(ts_i, np.ndarray):
                    ts_i = np.expand_dims(ts_i, axis=1)
                else:
                    ts_i = ts_i.unsqueeze(1)
            means_i, vars_i = self.compute_single_ts_means_and_vars(
                ts_i, mean_0, var_0, alpha_mean, alpha_var
            )
            self.means.append(means_i)
            self.vars.append(vars_i)
        return

    def neg_log_likelihood_Gaussian(
        self,
        ts: np.ndarray | Tensor,
        mean_0: Tensor,
        var_0: Tensor,
        alpha_mean: Tensor,
        alpha_var: Tensor,
    ) -> float:
        """
        This method computes the function that must be minimized to find the optimal
        values of the static parameters and the initial values of mean and var.
        """
        if isinstance(ts, np.ndarray):
            ts = torch.from_numpy(ts)

        len_ts = ts.shape[0]

        mean_ts = torch.zeros(len_ts)
        var_ts = torch.zeros(len_ts)
        log_likelihood_ts = torch.zeros(len_ts)
        """y = np.append(y, y[T - 1])"""

        mean, var = mean_0, var_0
        for i, ts_i in enumerate(ts):
            mean, var = self.update_mean_and_var(ts_i, mean, var, alpha_mean, alpha_var)
            mean_ts[i] = mean
            var_ts[i] = var
            log_likelihood_ts[i] = -0.5 * torch.log(
                2 * torch.pi * var + self.eps
            ) - 0.5 * (ts_i - mean) ** 2 / (var + self.eps)

        neg_log_lokelihood = -torch.sum(log_likelihood_ts).item()
        return neg_log_lokelihood / len_ts

    def compute_static_parameters(self, ts: TsDataset) -> None:
        # define initial guesses and bounds for the minimization
        initial_guesses = np.array([0.001, 0.001, 0, 1])
        bounds = ((None, None), (0.00001, 1), (0, 1), (0, 1))

        # define minimization function
        def minimization_funct(ts, mean_0, var_0, alpha_mean, alpha_var):
            return self.neg_log_likelihood_Gaussian(
                ts, mean_0, var_0, alpha_mean, alpha_var
            )

        for i, ts_i in enumerate(ts):
            print(f"Finding parameters for time series {i} of {len(ts)}")
            optimal = minimize(
                lambda params: minimization_funct(ts_i, *params),
                x0=initial_guesses,
                bounds=bounds,
            )
            alpha_mean, alpha_sigma, mean_0, sigma2_0 = optimal.x
            self.alpha_means.append(torch.tensor(alpha_mean).reshape(-1))
            self.alpha_vars.append(torch.tensor(alpha_sigma).reshape(-1))
            self.means_0.append(torch.tensor(mean_0).reshape(-1))
            self.vars_0.append(torch.tensor(sigma2_0).reshape(-1))

        return
'''


class GASComplexGaussian:
    def __init__(
        self,
        eps: float = 1e-9,
        regularization: str = "full",
    ) -> None:
        self.eps = eps
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

    def neg_log_likelihood_Gaussian(
        self,
        ts: np.ndarray,
        mean_0: np.ndarray,
        var_0: np.ndarray,
        alpha_mean: float,
        alpha_var: float,
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
            mean, var = self.update_mean_and_var(ts_i, mean, var, alpha_mean, alpha_var)
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
        alpha_mean = x[2 * n_features]
        alpha_var = x[2 * n_features + 1]
        return mean_0, var_0, alpha_mean, alpha_var

    def warm_up(
        self,
        dataset: list[np.ndarray],
        initial_guesses: np.ndarray = np.array([0.001, 0.001, 0, 1], dtype="float"),
        bounds: tuple = ((None, None), (0.00001, 1), (0, 0.999), (0, 0.999)),
    ) -> list[dict]:
        """
        This method computes the ideal initial guesses and static parameters for each of the input time series in the list.
        Ideal results are obtained as minimizers of the negative log likelihood function.
        It returns the initial guesses and static parameters as lists of numpy arrays.
        """
        n_features = 1 if len(dataset[0].shape) == 1 else dataset[0].shape[1]
        # let's check that initial guesses and bounds are of the correct shape
        assert (
            initial_guesses.shape[0] == 2 * n_features + 2
        ), "initial_guesses must be a 1D array of shape (2 * n_features + 2, ). First n_features elements are the means of each feature, second n_features elements are the vars of each feature, last two elements are alpha_mean and alpha_var"
        assert (
            len(bounds) == 2 * n_features + 2
        ), "bounds must be a tuple of 2 * n_features + 2 elements (pair of values). First n_features elements are bounds for means of each feature, second n_features elements are bounds for vars of each feature, last two elements are bounds for alpha_mean and alpha_var"

        initial_params_list = []
        for ts in tqdm(dataset, total=len(dataset), unit="ts"):
            # quick correction of shapes. We always assume (length, n_features)
            if len(ts.shape) == 1:
                ts = np.expand_dims(ts, axis=1)

            # define minimization function. It takes a 1D np.ndarray as input
            # the shape of this input is (4 * n_features, ) containing the initial
            # guesses for each ts feature and for each of the 4 parameter of the
            # normalizer
            def func_to_minimize(x):
                # we must first unpack the input
                mean_0, var_0, alpha_mean, alpha_var = self.unpack_minimization_input(
                    x, n_features
                )
                return self.neg_log_likelihood_Gaussian(
                    ts, mean_0, var_0, alpha_mean, alpha_var
                )

            optimal = minimize(
                lambda x: func_to_minimize(x),
                x0=initial_guesses,
                bounds=bounds,
            )
            mean_0, var_0, alpha_mean, alpha_sigma = self.unpack_minimization_input(
                optimal.x, n_features
            )

            initial_params_list.append(
                {
                    "alpha_mean": alpha_mean,
                    "alpha_sigma": alpha_sigma,
                    "mean_0": mean_0,
                    "var_0": var_0,
                }
            )

        return initial_params_list

    def normalize(
        self, dataset: list[np.ndarray], normalizer_params: list[dict]
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """
        This method normalizer a dataset (list) of time series. It needs also time
        series parameters, which are the output of the warm_up method. It returns
        the dataset of normalized time series, the means and the vars of each time
        series. Each element of the list will be a 1D or a 2D tensor, depending
        on the input.
        """
        norm_dataset, means, vars = [], [], []
        for ts, ts_params in zip(dataset, normalizer_params):
            is_univariate = len(ts.shape) == 1
            # we must expand dims because computations are performed on 2D arrays
            if is_univariate:
                ts = np.expand_dims(ts, axis=1)

            ts_means = np.empty_like(ts)
            ts_vars = np.empty_like(ts)

            mean = ts_params["mean_0"]  # (n_features)
            var = ts_params["var_0"]  # (n_features)
            alpha_mean = ts_params["alpha_mean"]  # float
            alpha_var = ts_params["alpha_sigma"]  # float

            for i, ts_i in enumerate(ts):
                mean, var = self.update_mean_and_var(
                    ts_i, mean, var, alpha_mean, alpha_var
                )
                ts_means[i] = mean
                ts_vars[i] = var
            norm_ts = (ts - ts_means) / (np.sqrt(ts_vars) + self.eps)

            if is_univariate:
                norm_ts = norm_ts.squeeze(1)
                ts_means = ts_means.squeeze(1)
                ts_vars = ts_vars.squeeze(1)

            norm_dataset.append(norm_ts)
            means.append(ts_means)
            vars.append(ts_vars)
        return norm_dataset, means, vars
