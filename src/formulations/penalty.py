from typing import Optional

import torch

from .formulation import Formulation


class KthOrderPenaltyFormulation(Formulation):
    def __init__(self, order: int, penalty_coefficient: float, penalty_gamma: float, *args, **kwargs):
        assert order == 2
        assert penalty_coefficient > 0
        assert penalty_gamma > 1

        # TODO(juan43ramirez): which order values are principled?
        self.order = order
        self.penalty_coefficient = penalty_coefficient
        self.penalty_gamma = penalty_gamma

        super().__init__(*args, **kwargs)

    def compute_lagrangian(
        self,
        loss: torch.Tensor,
        ineq_constraints: Optional[tuple[torch.Tensor]] = None,
        eq_constraints: Optional[tuple[torch.Tensor]] = None,
        *args,
        **kwargs,
    ):
        lagrangian = torch.clone(loss)

        if ineq_constraints is not None:
            lagrangian += torch.sum(self.penalty_coefficient * ineq_constraints.relu().pow(self.order)) / 2

        if eq_constraints is not None:
            lagrangian += torch.sum(self.penalty_coefficient * eq_constraints.abs().pow(self.order)) / 2

        return lagrangian

    @torch.no_grad()
    def update_state_(self, *args, **kwargs):

        # TODO: only update if constraint violation does not decrease
        self.penalty_coefficient *= self.penalty_gamma

    @property
    def update_multipliers_on_step(self):
        """Penalized formulations are a sequence of unconstrained problems. Penalty
        coefficients are not necessarily updated on each step."""
        return False

    def update_multipliers_(*args, **kwargs):
        pass


def QuadraticPenaltyFormulation(*args, **kwargs):
    return KthOrderPenaltyFormulation(order=2, *args, **kwargs)
