import abc
from typing import Optional

import torch


class Formulation(abc.ABC):
    def __init__(self, *args, **kwargs):
        pass

    def compute_lagrangian(
        self,
        loss: torch.Tensor,
        ineq_constraints: Optional[tuple[torch.Tensor]] = None,
        eq_constraints: Optional[tuple[torch.Tensor]] = None,
        ineq_multipliers: Optional[tuple[torch.Tensor]] = None,
        eq_multipliers: Optional[tuple[torch.Tensor]] = None,
        *args,
        **kwargs,
    ):
        pass

    def update_multipliers(*args, **kwargs):
        pass
