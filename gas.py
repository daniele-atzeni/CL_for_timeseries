import torch
from torch import Tensor
import torch.nn as nn


class GASCell(nn.Module):
    def __init__(
        self,
        eta_mu: float,
        eta_var: float,
        eps: float = 1e-13,
    ) -> None:
        super(GASCell, self).__init__()
        self.eta_mu = eta_mu
        self.eta_var = eta_var
        self.eps = eps  # this is to avoid division by zero

    def update_mu_var(
        self, x: Tensor, mu: Tensor, var: Tensor
    ) -> tuple[Tensor, Tensor]:
        # update mu and var
        mu = mu + self.eta_mu * (x - mu)
        var = var * (1 - self.eta_var) + self.eta_var * (x - mu) ** 2
        return mu, var

    def forward(
        self, x: Tensor, mu: Tensor, var: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        # check input: expected tensor of shape (n_features)
        if x.dim() != 1:
            raise ValueError(f"Wrong number of dimensions. Expected 1, got {x.dim()}.")
        # update
        mu, var = self.update_mu_var(x, mu, var)
        # normalize input
        norm_x = (x - mu) / (torch.sqrt(var) + self.eps)

        return norm_x, mu, var


class GASLayer(nn.Module):
    def __init__(
        self,
        eta_mu: float,
        eta_var: float,
    ) -> None:
        super(GASLayer, self).__init__()
        self.gas_cell = GASCell(eta_mu, eta_var)

    def init_mu_var(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # initialize them as mean and variance of the whole time series
        # x is assumed (ts_length, n_features)
        # # we can try other methods too
        mu = torch.mean(x, dim=0)
        var = torch.var(x, dim=0)
        return mu, var

    def forward(
        self, x: Tensor, mu: float | None = None, var: float | None = None
    ) -> tuple[Tensor, Tensor]:
        # check input
        input_dim = x.dim()
        if not (input_dim == 2 or input_dim == 3):
            raise ValueError(
                f"Wrong number of dimensions. Expected 2 or 3, got {input_dim}."
            )
        # assumed tensor of shape (ts_length, n_features) or (1, ts_length, n_features)
        if input_dim == 3:
            if x.shape[0] != 1:
                raise ValueError("This method supports only online learning.")
            x = x.squeeze(0)

        # initialize mu and vars of the gas cell if they are None
        if mu is None or var is None:
            assert (
                mu is None and var is None
            ), "You must pass either mu and var or none of them"
            mu, var = self.init_mu_var(x)

        # initialize results
        norm_x = torch.empty_like(x)
        mus = torch.empty_like(x)
        vars = torch.empty_like(x)

        for i, x_i in enumerate(x):
            norm_x_i, mu, var = self.gas_cell(x_i, mu, var)
            # save results
            norm_x[i], mus[i], vars[i] = norm_x_i, mu, var

        # return tensor of the same shape as original input
        if input_dim == 3:
            norm_x = norm_x.unsqueeze(0)
            mus = mus.unsqueeze(0)
            vars = vars.unsqueeze(0)

        # combine additional information into one single tensor of shape (1, ts_length, 2*n_features)
        additional_info = torch.cat((mus, vars), dim=-1)

        return norm_x, additional_info


class GASNormLayer(nn.Module):
    def __init__(self, eta_mu: float, eta_var: float, ts_processer: nn.Module) -> None:
        super(GASNormLayer, self).__init__()

        self.gas_layer = GASLayer(eta_mu, eta_var)
        self.ts_processer = ts_processer

    def forward(
        self, x: Tensor, mu: float | None = None, var: float | None = None
    ) -> Tensor:
        if x.dim() != 3:
            raise ValueError(f"Wrong number of dimensions. Expected 3, got {x.dim()}.")

        # only online mode supported 'til now
        if x.shape[0] != 1:
            raise ValueError("This method supports only online learning.")

        norm_x, add_info = self.gas(x)
        processed_norm_x = self.ts_processer(norm_x)
        mus, vars = add_info[:, : x.shape[1]], add_info[:, x.shape[1] :]

        return vars * (processed_norm_x + mus)


class GASModel(nn.Module):
    def __init__(
        self,
        ts_encoder: nn.Module,  # model to embed time series data
        eta_mu: float,
        eta_var: float,
        output_model: nn.Module,  # downstream model
    ) -> None:
        super(GASModel, self).__init__()

        self.gas = GASLayer(eta_mu, eta_var)
        self.ts_encoder = ts_encoder
        self.output_model = output_model

    def forward(self, x: Tensor) -> Tensor:
        # assuming shape (batch, ts_length, n_features)
        if x.dim() != 3:
            raise ValueError(f"Wrong number of dimensions. Expected 3, got {x.dim()}.")

        # only online mode supported 'til now
        if x.shape[0] != 1:
            raise ValueError("This method supports only online learning.")

        x, add_info = self.gas(x)
        # tensors shape (1, ts_length, n_features) and (1, ts_length, 2*n_features)

        ####################### possibly a lot of new features!!!
        add_info = add_info.reshape(x.shape[0], -1)
        # this becomes (batch, ts_length * 2 * n_features)
        #######################

        # process the normalized timeseries
        x = self.ts_encoder(x)  # shape (batch, n_encoded_features)
        # check that the output_model has the correct input shape
        # output_in_dim = self.output_model.input_shape
        # log_sentence = f'Input n_feat of the output model must be encoded_dim + (ts_length * n_features), got {output_in_dim}.'
        # assert output_in_dim == (x.shape[0], x.shape[1] + mus.shape[1]), log_sentence

        # concatenate normalized x and the means
        x = torch.cat((x, add_info), dim=1)

        return self.output_model(x)
