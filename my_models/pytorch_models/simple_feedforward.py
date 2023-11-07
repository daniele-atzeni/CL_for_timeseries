import torch


class FFNN(torch.nn.Module):
    def __init__(
        self,
        mean_layer: torch.nn.Linear,
        n_features: int,
        context_length: int,
        prediction_length: int,
        num_hidden_dimensions: list[int],
    ):
        super(FFNN, self).__init__()

        self.mean_layer = mean_layer

        hidden_layers = []
        # first layer
        hidden_layers.append(
            torch.nn.Linear(n_features * context_length, num_hidden_dimensions[0])
        )
        hidden_layers.append(torch.nn.ReLU())
        for i, hidden_dim in enumerate(num_hidden_dimensions[:-1]):
            hidden_layers.append(
                torch.nn.Linear(hidden_dim, num_hidden_dimensions[i + 1])
            )
            hidden_layers.append(torch.nn.ReLU())
        # last layer
        hidden_layers.append(
            torch.nn.Linear(num_hidden_dimensions[-1], prediction_length * n_features)
        )

        self.layers = torch.nn.Sequential(*hidden_layers)

    def forward(self, x, mean):
        batch, _, n_features = x.shape
        x = x.reshape(batch, -1)  # flatten
        out_layers = self.layers(x)
        out_mean_layer = self.mean_layer(mean)
        out = out_layers + out_mean_layer
        return out.reshape(batch, -1, n_features)
