from typing import Optional

import torch

from .formulation import Formulation


class ExactPenaltyFormulation(Formulation):
    def __init__(self, penalty_coefficient: float, penalty_gamma: float, *args, **kwargs):
        assert penalty_coefficient > 0
        assert penalty_gamma > 1

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
            lagrangian += self.penalty_coefficient * ineq_constraints.relu().max()

        if eq_constraints is not None:
            lagrangian += self.penalty_coefficient * eq_constraints.abs().max()

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
