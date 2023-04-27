import numpy as np
import torch


class MLP(torch.nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=None):
        super(MLP, self).__init__()

        self.hidden_dims = hidden_dims

        layers = torch.nn.ModuleList()
        in_dim = np.prod(input_dim)
        for h_dim in hidden_dims:
            layers.append(torch.nn.Linear(in_dim, h_dim))
            layers.append(torch.nn.ReLU())
            in_dim = h_dim

        layers.append(torch.nn.Linear(in_dim, num_classes))
        self.layers = torch.nn.Sequential(*layers)

    @property
    def num_layers(self):
        return len(self.hidden_dims) + 1 if self.hidden_dims is not None else 1

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.layers(x)


def LogisticRegression(input_dim, num_classes):
    return MLP(input_dim, num_classes, hidden_dims=[])
