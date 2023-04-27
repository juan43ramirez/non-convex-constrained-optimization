"""Code inspired in Cooper. See https://github.com/cooper-org/cooper/."""

import torch


class Multiplier(torch.nn.Module):
    def __init__(self, init: torch.Tensor, enforce_positive: bool = False):
        super().__init__()

        self.enforce_positive = enforce_positive
        self.weight = torch.nn.Parameter(init)
        self.device = self.weight.device

    @property
    def shape(self):
        return self.weight.shape

    def post_step_(self):
        if self.enforce_positive:
            # Ensures non-negativity for multipliers associated with inequality constraints.
            self.weight.data = torch.relu(self.weight.data)

    def state_dict(self):
        _state_dict = super().state_dict()
        _state_dict["enforce_positive"] = self.enforce_positive
        return _state_dict

    def load_state_dict(self, state_dict):
        self.enforce_positive = state_dict.pop("enforce_positive")
        super().load_state_dict(state_dict)
        self.device = self.weight.device

    def forward(self):
        return self.weight
