import torch
from torch import Tensor
import torch.nn as nn

# if we want to concatenate, the denormalizer must know various shapes of the input and the output of the model


class Denormalizer(torch.nn.Module):
    """
    Denormalizer Interface
    We assume each denormalizer composed of:
    - mus encoders, i.e. some kind of function that process means information from the normalization phase
    - vars encoders, i.e. some kind of function that process vars information from the normalization phase
    - a denormalize method that describes how to denormalize the output of the model
    """

    def __init__(self) -> None:
        super(Denormalizer, self).__init__()

    def forward(self, x: Tensor, mus: Tensor, vars: Tensor) -> Tensor:
        return self.denormalize(x, mus, vars)

    def denormalize(self, x: Tensor, mus: Tensor, vars: Tensor) -> Tensor:
        raise NotImplementedError()


class ConcatDenormalizer(Denormalizer):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(ConcatDenormalizer, self).__init__()
        self.output_layer = nn.Linear(input_dim, output_dim)

    def denormalize(self, x: Tensor, mus: Tensor, vars: Tensor) -> Tensor:
        # x tensor of shape (batch, embed_dim)
        # mus and vars can be
        # - shape (batch, ts_length, n_features) for GAS normalizer
        # - shape (batch, n_features) for Batch norm
        assert (
            mus.shape == vars.shape
        ), f"Something went wrong, means and vars have different shapes. Respectively {mus.shape} and {vars.shape}"
        assert mus.dim() in {2, 3}, f"Unknown mus and vars shape: {mus.shape}"
        if mus.dim() == 3:
            mus = mus.reshape((x.shape[0], -1))
            vars = vars.reshape((x.shape[0], -1))
        cat_tensor = torch.cat((x, mus, vars), dim=1)
        return self.output_layer(cat_tensor)


class SumDenormalizer(Denormalizer):
    """
    We assume that our output function is in the form y = f(x)g(sigma) + h(mu)
    we will learn this three functions separately.
    The first one we learn is h and assume it is a linear function (NB this is robust to magnitude changes!)
    The second one is f and it will be the main ML model
    The third one is g and it will be a linear function again (NB this is robust to magnitude changes!)
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(SumDenormalizer, self).__init__()
        self.mus_encoder = nn.Linear(input_dim, output_dim)
        self.vars_encoder = nn.Linear(input_dim, output_dim)

    def forward(
        self,
        proc_x: Tensor | None,
        mus: Tensor,
        vars: Tensor | None,
    ) -> Tensor:
        proc_mus = self.process_mus(mus)
        if proc_x is None:
            proc_x = torch.zeros_like(proc_mus)
        if vars is None:
            proc_vars = torch.zeros_like(proc_mus)
        else:
            proc_vars = self.process_vars(vars)
        return proc_x * proc_vars + proc_mus

    def process_mus(self, mus: Tensor) -> Tensor:
        # mus are expected as (batch, ts_length, n_features)
        batch, _, n_features = mus.shape
        mus = mus.reshape((mus.shape[0], -1))
        return self.mus_encoder(mus).reshape((batch, -1, n_features))

    def process_vars(self, vars: Tensor) -> Tensor:
        # vars are expected as (batch, ts_length, n_features)
        batch, _, n_features = vars.shape
        vars = vars.reshape((vars.shape[0], -1))
        return self.vars_encoder(vars).reshape((batch, -1, n_features))

    def get_mus_params(self):
        return self.mus_encoder.parameters()

    def get_vars_params(self):
        return self.vars_encoder.parameters()
