import torch
from torch import Tensor
import torch.nn as nn


class Normalizer(nn.Module):
    """
    Define the Normalizer interface. This interface assumes that inputs are always
    composed of two parts:
    - pieces of a multivariate time series
    - indices of the pieces in the original time series
    """

    # indices are useful for GAS normalizers, so we can precompute
    # means and vars time series and retrieve corresponding pieces

    def __init__(self) -> None:
        super(Normalizer, self).__init__()

    def warm_up(self, *args, **kwargs) -> None:
        pass

    def forward(self, ts_indices: Tensor, ts: Tensor) -> Tensor:
        return self.normalize(ts_indices, ts)

    def normalize(self, ts_indices: Tensor, ts: Tensor) -> Tensor:
        raise NotImplementedError()


class GASNormalizer(Normalizer):
    """
    Define GAS-based Normalizer interface
    """

    def __init__(self) -> None:
        super(GASNormalizer, self).__init__()

    def compute_parameters(self, ts: Tensor) -> None:
        raise NotImplementedError()

    def compute_mus_and_vars(self, ts: Tensor) -> None:
        raise NotImplementedError()

    def warm_up(self, train_ts: Tensor, complete_ts: Tensor) -> None:
        self.compute_parameters(train_ts)
        self.compute_mus_and_vars(complete_ts)
        return


class GASSimpleGaussian(GASNormalizer):
    def __init__(self, eps: float = 1e-9) -> None:
        super(GASSimpleGaussian, self).__init__()
        self.eta_mu = None
        self.eta_var = None
        self.eps = eps

        self.mus = None
        self.vars = None

    def compute_parameters(self, train_ts: Tensor) -> None:
        self.eta_mu = 0.5
        self.eta_var = 0.5
        return

    def update_mu_var(
        self, ts_i: Tensor, mu: Tensor | float, var: Tensor | float
    ) -> tuple[Tensor | float, Tensor | float]:
        if self.eta_var is None or self.eta_mu is None:
            raise ValueError(
                "Something went wrong with the warm up, static parameters are not initialized."
            )
        # ts_i is the time series single element (shape = (n_features))
        # same shape for mu and var
        mu = mu + self.eta_mu * (ts_i - mu)
        var = var * (1 - self.eta_var) + self.eta_var * (ts_i - mu) ** 2
        return mu, var

    def compute_mus_and_vars(self, ts: Tensor) -> None:
        # ts is the complete time series (shape = (total_length, n_features))
        # we want to compute mus and vars of the same shape
        if ts.dim() == 1:
            ts = ts.unsqueeze(1)
        # intialize the results
        self.mus = torch.empty_like(ts)
        self.vars = torch.empty_like(ts)
        # intialize first mean and var
        mu = torch.mean(ts, dim=0)
        var = torch.var(ts, dim=0)
        self.mus[0] = mu
        self.vars[0] = var
        # compute mus and vars
        for i, ts_i in enumerate(ts):
            mu, var = self.update_mu_var(ts_i, mu, var)
            self.mus[i] = mu
            self.vars[i] = var

        return

    def normalize(
        self, ts_indices: Tensor, ts: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        # this is the forward pass
        # ts is shape (batch, ts_length, n_features) #### assuming BATCH_FIRST
        # ts_indices are the indices of time series pieces, shape = (batch, ts_length)
        # we must retrieve means and variances previously computed and normalize
        if self.mus is None or self.vars is None:
            raise ValueError(
                "Something went wrong with the computation of mus and vars, they are not initialized."
            )
        mus = torch.empty_like(ts)
        vars = torch.empty_like(ts)
        for i, ts_ind in enumerate(ts_indices):
            mus[i] = self.mus[ts_ind]
            vars[i] = self.vars[ts_ind]
        return (ts - mus) / (torch.sqrt(vars) + self.eps), mus, vars
