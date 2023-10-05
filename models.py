import torch
from torch import Tensor
from torch import nn


class MyModel(nn.Module):
    def __init__(self) -> None:
        super(MyModel, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


class FFNN(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        hid_dim_1: int,
        hid_dim_2: int,
        activation: str,
        output_dim: int,
    ) -> None:
        super(FFNN, self).__init__()
        if activation == "relu":
            self.activation = torch.nn.ReLU()
        else:
            self.activation = torch.nn.Tanh()
        self.hid_1 = torch.nn.Linear(inp_dim, hid_dim_1)
        self.hid_2 = torch.nn.Linear(hid_dim_1, hid_dim_2)
        self.out_layer = torch.nn.Linear(hid_dim_2, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        # x is a ts, shape (batch, ts_len, n_features)    # ASSUMING BATCH_FIRST
        x = x.reshape((x.shape[0], -1))
        x = self.activation(self.hid_1(x))
        x = self.activation(self.hid_2(x))
        x = self.out_layer(x)
        return x
